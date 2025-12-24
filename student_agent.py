"""
Inference agent that loads the trained Q-table when possible and falls back to
safe heuristics on unseen states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import pickle
import random

import numpy as np

from training_custom_taxi_env import TrainingEnvConfig


Action = int
Obs = Tuple[int, ...]
StateKey = Tuple
QTable = Dict[StateKey, np.ndarray]

# Reuse the training config for bucket logic and fuel defaults.
ENV_CFG = TrainingEnvConfig()
FUEL_LIMIT = getattr(ENV_CFG, "fuel_limit", 5000)
GRID_BUCKET_EDGES = ENV_CFG.grid_size_bucket_edges
FUEL_BUCKET_COUNT = ENV_CFG.fuel_bucket_count

# Globals for module-level state across steps.
RNG = random.Random(0)
Q_TABLE: QTable = {}
Q_META: Dict = {}
CHECKPOINT_PATH = Path("q_table.pkl")


def load_q_table(path: Path) -> Tuple[QTable, Dict]:
    if not path.exists():
        return {}, {}
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "q_table" in payload:
        table_raw = payload["q_table"]
        meta = payload.get("meta", {})
    else:
        table_raw = payload
        meta = {}
    table: QTable = {}
    for k, v in table_raw.items():
        table[tuple(k)] = np.array(v, dtype=np.float32)
    return table, meta


Q_TABLE, Q_META = load_q_table(CHECKPOINT_PATH)
USE_Q_TABLE = bool(Q_TABLE) and Q_META.get("mean_reward", 0) > 0


@dataclass
class AgentMemory:
    has_passenger: bool = False
    last_known_passenger_loc: Optional[Tuple[int, int]] = None
    dest_guess: Optional[Tuple[int, int]] = None
    station_list: Tuple[Tuple[int, int], ...] = ()
    search_idx: int = 0
    drop_idx: int = 0
    pickup_station: Optional[Tuple[int, int]] = None
    tried_pickups: Set[Tuple[int, int]] = field(default_factory=set)
    tried_dropoffs: Set[Tuple[int, int]] = field(default_factory=set)
    fuel: int = FUEL_LIMIT
    step_count: int = 0
    prev_obs: Optional[Obs] = None
    prev_action: Optional[Action] = None

    def reset(self) -> None:
        self.has_passenger = False
        self.last_known_passenger_loc = None
        self.dest_guess = None
        self.station_list = ()
        self.search_idx = 0
        self.drop_idx = 0
        self.pickup_station = None
        self.tried_pickups = set()
        self.tried_dropoffs = set()
        self.fuel = FUEL_LIMIT
        self.step_count = 0
        self.prev_obs = None
        self.prev_action = None

    def update_after_step(self, current_obs: Obs) -> None:
        """
        Update internal flags based on the last action/obs transition.
        Called at the start of get_action with the new observation.
        """
        if self.prev_obs is None or self.prev_action is None:
            return

        self.step_count += 1
        if self.fuel > 0:
            self.fuel -= 1

        prev_passenger_here = bool(self.prev_obs[14])
        prev_destination_here = bool(self.prev_obs[15])
        if self.station_list:
            self.station_list = tuple(self.station_list)

        if self.prev_action == 4 and prev_passenger_here:
            self.has_passenger = True
            self.last_known_passenger_loc = None
            self.pickup_station = (self.prev_obs[0], self.prev_obs[1])
            self.drop_idx = 0
            self.tried_dropoffs.clear()
        elif self.prev_action == 4 and not self.has_passenger and self.station_list:
            self.search_idx = (self.search_idx + 1) % len(self.station_list)
            self.tried_pickups.add((self.prev_obs[0], self.prev_obs[1]))

        if self.prev_action == 5 and self.has_passenger and prev_destination_here:
            self.has_passenger = False
            self.pickup_station = None
            self.drop_idx = 0
            self.tried_pickups.clear()
            self.tried_dropoffs.clear()
        elif self.prev_action == 5 and self.has_passenger:
            self.drop_idx = (self.drop_idx + 1) % max(1, len(self.station_list) - 1 if self.station_list else 1)
            self.tried_dropoffs.add((self.prev_obs[0], self.prev_obs[1]))

        # Track passenger location when seen and not in taxi.
        if not self.has_passenger and prev_passenger_here:
            taxi_row, taxi_col = self.prev_obs[0], self.prev_obs[1]
            self.last_known_passenger_loc = (taxi_row, taxi_col)

        # Destination guess: if we ever stand on it, remember.
        if prev_destination_here:
            taxi_row, taxi_col = self.prev_obs[0], self.prev_obs[1]
            self.dest_guess = (taxi_row, taxi_col)


MEMORY = AgentMemory()


def bucket_value(value: int, edges) -> int:
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def fuel_bucket(fuel: int) -> int:
    bucket_size = max(1, FUEL_LIMIT // max(1, FUEL_BUCKET_COUNT))
    idx = fuel // bucket_size
    return min(idx, max(0, FUEL_BUCKET_COUNT - 1))


def estimate_grid_size(obs: Obs) -> int:
    coords = obs[:10]  # taxi position + 4 station coords
    max_coord = max(coords)
    return max_coord + 1


def build_state_key(obs: Obs) -> StateKey:
    taxi_row, taxi_col = obs[0], obs[1]
    stations = (
        (obs[2], obs[3]),
        (obs[4], obs[5]),
        (obs[6], obs[7]),
        (obs[8], obs[9]),
    )
    MEMORY.station_list = stations
    obstacle_flags = obs[10:14]
    passenger_here = bool(obs[14])
    destination_here = bool(obs[15])

    # Update hints for location tracking.
    if not MEMORY.has_passenger and passenger_here:
        MEMORY.last_known_passenger_loc = (taxi_row, taxi_col)
    if destination_here:
        MEMORY.dest_guess = (taxi_row, taxi_col)

    passenger_state: Tuple[int, int] | str
    if MEMORY.has_passenger:
        passenger_state = "in_taxi"
    elif MEMORY.last_known_passenger_loc is not None:
        passenger_state = MEMORY.last_known_passenger_loc
    else:
        passenger_state = "unknown"

    if MEMORY.dest_guess is None:
        # Default guess to first station to keep keys stable.
        MEMORY.dest_guess = stations[0]

    grid_bucket = bucket_value(estimate_grid_size(obs), GRID_BUCKET_EDGES)
    fuel_b = fuel_bucket(MEMORY.fuel)

    return (
        taxi_row,
        taxi_col,
        passenger_state,
        MEMORY.dest_guess[0],
        MEMORY.dest_guess[1],
        *obstacle_flags,
        grid_bucket,
        fuel_b,
    )


def select_from_q(key: StateKey) -> Optional[Action]:
    q_values = Q_TABLE.get(key)
    if q_values is None:
        return None
    max_q = float(np.max(q_values))
    best_actions = np.flatnonzero(q_values == max_q)
    if len(best_actions) == 0:
        return None
    return int(RNG.choice(best_actions))


def heuristic_action(obs: Obs) -> Action:
    obstacle_n, obstacle_s, obstacle_e, obstacle_w = obs[10:14]
    taxi_pos = (obs[0], obs[1])
    stations = MEMORY.station_list or (
        (obs[2], obs[3]),
        (obs[4], obs[5]),
        (obs[6], obs[7]),
        (obs[8], obs[9]),
    )
    MEMORY.station_list = stations

    if not stations:
        return RNG.randrange(6)

    def nearest(candidates: Tuple[Tuple[int, int], ...]) -> Tuple[int, int]:
        return min(candidates, key=lambda c: abs(c[0] - taxi_pos[0]) + abs(c[1] - taxi_pos[1]))

    if not MEMORY.has_passenger:
        remaining = tuple(s for s in stations if s not in MEMORY.tried_pickups) or stations
        target = nearest(remaining)
        if taxi_pos == target:
            return 4  # Attempt pickup at this station
    else:
        drop_candidates = tuple(s for s in stations if s != MEMORY.pickup_station) or stations
        remaining_drop = tuple(s for s in drop_candidates if s not in MEMORY.tried_dropoffs) or drop_candidates
        target = nearest(remaining_drop)
        if taxi_pos == target:
            return 5  # Attempt dropoff at this candidate

    target_row, target_col = target
    taxi_row, taxi_col = taxi_pos

    # Move toward target while avoiding known obstacles.
    candidate_moves = []
    if target_row > taxi_row and not obstacle_s:
        candidate_moves.append(0)  # South
    if target_row < taxi_row and not obstacle_n:
        candidate_moves.append(1)  # North
    if target_col > taxi_col and not obstacle_e:
        candidate_moves.append(2)  # East
    if target_col < taxi_col and not obstacle_w:
        candidate_moves.append(3)  # West

    # If primary directions are blocked, try any open direction to keep exploring.
    if not candidate_moves:
        if not obstacle_s:
            candidate_moves.append(0)
        if not obstacle_n:
            candidate_moves.append(1)
        if not obstacle_e:
            candidate_moves.append(2)
        if not obstacle_w:
            candidate_moves.append(3)

    if candidate_moves:
        return RNG.choice(candidate_moves)
    return RNG.randrange(6)


def get_action(obs: Obs) -> Action:
    """
    Select an action given the current observation.
    Uses Q-table when a matching key exists; otherwise falls back to heuristics.
    """
    MEMORY.update_after_step(obs)
    state_key = build_state_key(obs)

    action = select_from_q(state_key) if USE_Q_TABLE else None
    if action is None:
        action = heuristic_action(obs)

    # Cache for the next step transition update.
    MEMORY.prev_obs = obs
    MEMORY.prev_action = action
    return action


def reset_agent_state() -> None:
    """Optional helper to reset internal tracking between episodes."""
    MEMORY.reset()
