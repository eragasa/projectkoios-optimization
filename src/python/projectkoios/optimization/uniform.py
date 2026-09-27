"""Deterministic bounded SplitMix64 uniform ask/tell optimizer."""

from __future__ import annotations

from ._serialization import (
    decode_feedback,
    decode_space,
    encode_feedback,
    encode_space,
    integer_value,
    list_value,
    required,
    string_value,
)
from ._validation import (
    MAX_U64,
    candidate_identifier,
    integer,
    run_identifier,
    u32,
    u64,
)
from .checkpoint import JsonValue, OptimizerCheckpoint
from .exceptions import (
    CheckpointError,
    DuplicateObservationError,
    OptimizerExhaustedError,
    UnknownCandidateError,
)
from .identity import OptimizerDescriptor, OptimizerIdentity
from .records import (
    Candidate,
    ObjectiveFeedback,
    OptimizerUpdate,
    ParameterSpace,
)


class BoundedUniformOptimizer:
    """Propose a finite deterministic uniform-box sequence.

    Calls to :meth:`ask` may create multiple outstanding candidates. Corresponding
    feedback may arrive in any order. Proposal order, not observation arrival order,
    breaks equal-loss ties.
    """

    __slots__ = (
        "_candidates",
        "_feedback",
        "_max_evaluations",
        "_next_proposal",
        "_observed_ids",
        "_outstanding",
        "_run_id",
        "_seed",
        "_space",
    )

    _DESCRIPTOR = OptimizerDescriptor(
        OptimizerIdentity("bounded-uniform", 1),
        "splitmix64-uniform-box:1",
        1,
    )

    def __init__(
        self,
        space: ParameterSpace,
        *,
        seed: int,
        max_evaluations: int,
        run_id: str,
    ) -> None:
        if type(space) is not ParameterSpace:
            raise TypeError("space must be ParameterSpace")
        normalized_seed = u64(seed, "seed")
        budget = u32(max_evaluations, "max_evaluations", allow_zero=False)
        normalized_run_id = run_identifier(run_id, budget - 1)
        self._space = space
        self._seed = normalized_seed
        self._max_evaluations = budget
        self._run_id = normalized_run_id
        self._next_proposal = 0
        self._candidates: dict[str, Candidate] = {}
        self._outstanding: dict[str, Candidate] = {}
        self._observed_ids: set[str] = set()
        self._feedback: list[ObjectiveFeedback] = []

    @property
    def descriptor(self) -> OptimizerDescriptor:
        """Return the exact version-one algorithm descriptor."""
        return self._DESCRIPTOR

    @property
    def run_id(self) -> str:
        """Return this optimizer-run identity."""
        return self._run_id

    @property
    def space(self) -> ParameterSpace:
        """Return the configured parameter space."""
        return self._space

    @property
    def feedback(self) -> tuple[ObjectiveFeedback, ...]:
        """Return feedback in accepted arrival order."""
        return tuple(self._feedback)

    @property
    def outstanding(self) -> tuple[Candidate, ...]:
        """Return outstanding candidates in proposal order."""
        return tuple(sorted(self._outstanding.values(), key=lambda item: item.sequence))

    @property
    def exhausted(self) -> bool:
        """Return whether every budgeted proposal has received feedback."""
        return self._next_proposal == self._max_evaluations and not self._outstanding

    def ask(self) -> Candidate:
        """Propose one bounded candidate or raise at proposal-budget exhaustion."""
        if self._next_proposal >= self._max_evaluations:
            raise OptimizerExhaustedError("uniform proposal budget is exhausted")
        sequence = self._next_proposal
        candidate = Candidate(
            candidate_identifier(self._run_id, sequence),
            self._run_id,
            sequence,
            self.sample_values(self._space, self._seed, sequence),
        )
        self._next_proposal += 1
        self._candidates[candidate.candidate_id] = candidate
        self._outstanding[candidate.candidate_id] = candidate
        return candidate

    def tell(self, feedback: ObjectiveFeedback, /) -> OptimizerUpdate:
        """Accept finite feedback for an outstanding candidate exactly once."""
        if type(feedback) is not ObjectiveFeedback:
            raise TypeError("feedback must be ObjectiveFeedback")
        if feedback.candidate_id in self._observed_ids:
            raise DuplicateObservationError("candidate feedback was already accepted")
        candidate = self._outstanding.get(feedback.candidate_id)
        if candidate is None:
            raise UnknownCandidateError("candidate is not outstanding for this run")
        del self._outstanding[feedback.candidate_id]
        self._observed_ids.add(feedback.candidate_id)
        self._feedback.append(feedback)
        best = self._best_candidate()
        termination = "evaluation_limit" if self.exhausted else None
        return OptimizerUpdate(
            feedback.candidate_id,
            len(self._feedback),
            best.candidate_id,
            termination is not None,
            termination,
        )

    def _best_candidate(self) -> Candidate:
        feedback_by_id = {value.candidate_id: value for value in self._feedback}
        return min(
            (self._candidates[candidate_id] for candidate_id in self._observed_ids),
            key=lambda candidate: (
                feedback_by_id[candidate.candidate_id].loss,
                candidate.sequence,
            ),
        )

    def checkpoint(self) -> OptimizerCheckpoint:
        """Return canonical replay-complete state."""
        payload: dict[str, JsonValue] = {
            "feedback": [encode_feedback(value) for value in self._feedback],
            "kind": "bounded-uniform",
            "max_evaluations": self._max_evaluations,
            "next_proposal": self._next_proposal,
            "run_id": self._run_id,
            "seed": self._seed,
            "space": encode_space(self._space),
        }
        return OptimizerCheckpoint.create(self.descriptor.identity.value, payload)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: OptimizerCheckpoint
    ) -> BoundedUniformOptimizer:
        """Restore state after strict schema, identity, and replay validation."""
        if type(checkpoint) is not OptimizerCheckpoint:
            raise TypeError("checkpoint must be OptimizerCheckpoint")
        if checkpoint.optimizer_identity != cls._DESCRIPTOR.identity.value:
            raise CheckpointError("checkpoint optimizer identity is incompatible")
        data = required(
            checkpoint.payload_object(),
            {
                "feedback",
                "kind",
                "max_evaluations",
                "next_proposal",
                "run_id",
                "seed",
                "space",
            },
            "uniform checkpoint",
        )
        if string_value(data["kind"], "kind") != "bounded-uniform":
            raise CheckpointError("checkpoint algorithm kind is incompatible")
        try:
            engine = cls(
                decode_space(data["space"]),
                seed=integer_value(data["seed"], "seed"),
                max_evaluations=integer_value(
                    data["max_evaluations"], "max_evaluations"
                ),
                run_id=string_value(data["run_id"], "run_id"),
            )
            proposal_count = integer_value(data["next_proposal"], "next_proposal")
            if not 0 <= proposal_count <= engine._max_evaluations:
                raise CheckpointError("next_proposal is outside the configured budget")
            for _ in range(proposal_count):
                engine.ask()
            for item in list_value(data["feedback"], "feedback"):
                engine.tell(decode_feedback(item))
        except (
            TypeError,
            ValueError,
            UnknownCandidateError,
            DuplicateObservationError,
        ) as error:
            raise CheckpointError("uniform checkpoint replay failed") from error
        return engine

    @staticmethod
    def unit_fraction(seed: int, counter: int) -> float:
        """Return one upper-53-bit binary64 SplitMix64 fraction in ``[0, 1)``."""
        normalized_seed = u64(seed, "seed")
        normalized_counter = integer(counter, "counter")
        if normalized_counter < 0:
            raise ValueError("counter must be nonnegative")
        word = (
            normalized_seed + 0x9E3779B97F4A7C15 * (normalized_counter + 1)
        ) & MAX_U64
        word = ((word ^ (word >> 30)) * 0xBF58476D1CE4E5B9) & MAX_U64
        word = ((word ^ (word >> 27)) * 0x94D049BB133111EB) & MAX_U64
        word ^= word >> 31
        return float(word >> 11) / 9_007_199_254_740_992.0

    @classmethod
    def sample_values(
        cls, space: ParameterSpace, seed: int, sample_index: int
    ) -> tuple[float, ...]:
        """Map one sample-major counter block into the configured closed bounds."""
        if type(space) is not ParameterSpace:
            raise TypeError("space must be ParameterSpace")
        normalized_index = integer(sample_index, "sample_index")
        if normalized_index < 0:
            raise ValueError("sample_index must be nonnegative")
        dimension = space.dimension
        return tuple(
            bound.lower
            + cls.unit_fraction(seed, normalized_index * dimension + index)
            * bound.width
            for index, bound in enumerate(space.bounds)
        )
