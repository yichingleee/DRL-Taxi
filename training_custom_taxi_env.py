from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, List
import random


@dataclass
class TrainingEnvConfig:
    """
    Easily-tunable settings for the training Taxi environment.
    Adjust ranges/penalties here instead of patching logic later.
    """
    grid_size_range: Tuple[int, int] = (5, 8)           # Environment Setup: randomize grid size (5–8+)
    obstacle_density_range: Tuple[float, float] = (0.05, 0.15)  # Environment Setup: 5–15% density
    fuel_limit: int = 5000                              # Mirror spec fuel cap

    dropoff_reward: float = 50.0                        # Reward Shaping: +50 drop-off
    step_penalty: float = 0.1                           # Reward Shaping: -0.1 per step
    invalid_action_penalty: float = 10.0                # Reward Shaping: -10 invalid pickup/drop
    obstacle_penalty: float = 5.0                       # Reward Shaping: -5 wall/obstacle
    fuel_depletion_penalty: float = 10.0                # Reward Shaping: -10 on fuel depletion

    # Optional light shaping to guide exploration
    enable_light_shaping: bool = True
    no_passenger_extra_penalty: float = 0.02            # Extra -0.02/step when not carrying passenger
    pickup_bonus: float = 0.5                           # Small +0.5 on successful pickup

    # Station control: None → randomize each episode; provide 4 coords to fix them.
    station_positions: Optional[Sequence[Tuple[int, int]]] = None

    # Seeding helpers for debugging (pass `seed` to reset(...) for per-episode seeds).
    base_seed: Optional[int] = None

    # Bucketing helpers for state-key construction
    fuel_bucket_count: int = 8                          # Tune to 5–10 buckets as suggested
    grid_size_bucket_edges: Sequence[int] = (5, 6, 7, 8, 9, 10)

    def __post_init__(self) -> None:
        min_grid, max_grid = self.grid_size_range
        if min_grid < 5 or min_grid > max_grid:
            raise ValueError("grid_size_range must be (>=5, >=min).")
        low_density, high_density = self.obstacle_density_range
        if not (0 <= low_density <= high_density <= 1):
            raise ValueError("obstacle_density_range must be within [0, 1] and ordered.")
        if self.station_positions is not None and len(self.station_positions) != 4:
            raise ValueError("station_positions must contain exactly 4 coordinates or be None.")


class TrainingCustomTaxiEnv:
    """
    Training-oriented Taxi environment with adjustable randomness and reward shaping.

    Observation format mirrors the simplified testing stub (simple_custom_taxi_env.py)
    to keep agent interfaces unchanged. Use get_state_key(...) for richer Q-table keys.
    """

    ACTIONS = {
        0: "Move South",
        1: "Move North",
        2: "Move East",
        3: "Move West",
        4: "Pick Up",
        5: "Drop Off",
    }

    def __init__(self, config: Optional[TrainingEnvConfig] = None):
        self.config = config or TrainingEnvConfig()
        self.rng = random.Random(self.config.base_seed)

        # Episode state
        self.grid_size: int = 0
        self.current_fuel: int = 0
        self.passenger_picked_up: bool = False
        self.taxi_pos: Tuple[int, int] = (0, 0)
        self.passenger_loc: Tuple[int, int] = (0, 0)
        self.destination: Tuple[int, int] = (0, 0)
        self.obstacles: set = set()
        self.stations: List[Tuple[int, int]] = []

    # --- Core API ---
    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[int, ...], Dict]:
        """
        Reset the environment.

        Args:
            seed: Optional per-episode seed for reproducible debugging runs.
        """
        if seed is not None:
            self.rng.seed(seed)

        self.grid_size = self.rng.randint(*self.config.grid_size_range)
        self.current_fuel = self.config.fuel_limit
        self.passenger_picked_up = False

        self._all_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        self.stations = self._sample_stations()
        self.obstacles = self._sample_obstacles()

        self.passenger_loc = self.rng.choice(self.stations)
        destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = self.rng.choice(destinations)

        self.taxi_pos = self._sample_taxi_position()

        obs = self._get_observation()
        info = {"grid_size": self.grid_size, "seed": seed}
        return obs, info

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict]:
        """Perform one step following the spec + light shaping."""
        reward = 0.0
        done = False

        # Movement
        next_row, next_col = self.taxi_pos
        if action == 0:   # South
            next_row += 1
        elif action == 1:  # North
            next_row -= 1
        elif action == 2:  # East
            next_col += 1
        elif action == 3:  # West
            next_col -= 1

        if action in (0, 1, 2, 3):
            blocked = (
                (next_row, next_col) in self.obstacles
                or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size)
            )
            if blocked:
                reward -= self.config.obstacle_penalty
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        elif action == 4:  # PICKUP
            if self.taxi_pos == self.passenger_loc and not self.passenger_picked_up:
                self.passenger_picked_up = True
                if self.config.enable_light_shaping:
                    reward += self.config.pickup_bonus
            else:
                reward -= self.config.invalid_action_penalty
        elif action == 5:  # DROPOFF
            if self.passenger_picked_up and self.taxi_pos == self.destination:
                reward += self.config.dropoff_reward
                done = True
                self.passenger_picked_up = False
            else:
                reward -= self.config.invalid_action_penalty
        else:
            raise ValueError(f"Invalid action {action}")

        # Step penalties and shaping applied regardless of success/failure.
        reward -= self.config.step_penalty
        if self.config.enable_light_shaping and not self.passenger_picked_up:
            reward -= self.config.no_passenger_extra_penalty

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            reward -= self.config.fuel_depletion_penalty
            done = True

        obs = self._get_observation()
        info = {"grid_size": self.grid_size}
        return obs, reward, done, info

    # --- Helpers for Q-table state encoding and debugging ---
    def get_state_key(self, obs: Optional[Tuple[int, ...]] = None) -> Tuple:
        """
        Construct a richer, bucketed state key as suggested in STRATEGY.md.
        Includes grid/fuel buckets to promote generalization.
        """
        obs = obs or self._get_observation()
        taxi_row, taxi_col = obs[0], obs[1]
        passenger_here = self.passenger_picked_up
        passenger_state = "in_taxi" if passenger_here else self.passenger_loc
        dest_row, dest_col = self.destination

        obstacle_flags = obs[10:14]  # N, S, E, W order preserved from observation
        grid_bucket = self._bucket_value(self.grid_size, self.config.grid_size_bucket_edges)
        fuel_bucket = self._fuel_bucket()

        return (
            taxi_row,
            taxi_col,
            passenger_state,
            dest_row,
            dest_col,
            *obstacle_flags,
            grid_bucket,
            fuel_bucket,
        )

    # --- Internal utilities ---
    def _sample_stations(self) -> List[Tuple[int, int]]:
        if self.config.station_positions is not None:
            return list(self.config.station_positions)
        return self.rng.sample(self._all_cells, 4)

    def _sample_obstacles(self) -> set:
        density = self.rng.uniform(*self.config.obstacle_density_range)
        target_count = int(round(density * self.grid_size * self.grid_size))
        forbidden = set(self.stations)
        candidates = [cell for cell in self._all_cells if cell not in forbidden]
        target_count = min(target_count, len(candidates))
        return set(self.rng.sample(candidates, target_count))

    def _sample_taxi_position(self) -> Tuple[int, int]:
        forbidden = set(self.obstacles) | set(self.stations) | {self.passenger_loc, self.destination}
        candidates = [cell for cell in self._all_cells if cell not in forbidden]
        if not candidates:
            raise RuntimeError("No available cell to place the taxi. Reduce obstacle density or grid size.")
        return self.rng.choice(candidates)

    def _get_observation(self) -> Tuple[int, ...]:
        taxi_row, taxi_col = self.taxi_pos

        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle

        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
        destination_look = (
            destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle
        )

        return (
            taxi_row,
            taxi_col,
            self.stations[0][0],
            self.stations[0][1],
            self.stations[1][0],
            self.stations[1][1],
            self.stations[2][0],
            self.stations[2][1],
            self.stations[3][0],
            self.stations[3][1],
            obstacle_north,
            obstacle_south,
            obstacle_east,
            obstacle_west,
            passenger_look,
            destination_look,
        )

    def _bucket_value(self, value: int, edges: Sequence[int]) -> int:
        # Returns the index of the first edge greater than value; capped to last bucket.
        for idx, edge in enumerate(edges):
            if value <= edge:
                return idx
        return len(edges)

    def _fuel_bucket(self) -> int:
        bucket_count = max(1, self.config.fuel_bucket_count)
        bucket_size = max(1, self.config.fuel_limit // bucket_count)
        bucket_idx = self.current_fuel // bucket_size
        return min(bucket_idx, bucket_count - 1)


if __name__ == "__main__":
    # Minimal smoke test to visualize defaults and easy tuning.
    env = TrainingCustomTaxiEnv()
    obs, info = env.reset(seed=42)  # Use per-episode seed for reproducible debugging runs.
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < 20:
        action = env.rng.choice(list(TrainingCustomTaxiEnv.ACTIONS.keys()))
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Finished episode in {steps} steps, reward={total_reward:.2f}, grid_size={info['grid_size']}")
