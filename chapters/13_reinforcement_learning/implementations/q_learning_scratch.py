import numpy as np

class QLearningAgent:
    """
    Tabular Q-Learning Agent for discrete environments[cite: 1].
    """
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha     # Learning rate
        self.gamma = gamma     # Discount factor
        self.epsilon = epsilon # Exploration rate
        
        # Initialize Q-table to zeros
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            return np.random.choice(self.n_actions) # Explore
        else:
            return int(np.argmax(self.q_table[state, :])) # Exploit

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Updates the Q-value using the Temporal Difference (TD) error[cite: 1].
        Equation: Q(s,a) <- Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s,a)]
        """
        best_next_q = 0.0 if done else np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        
        self.q_table[state, action] += self.alpha * td_error

if __name__ == "__main__":
    np.random.seed(42)
    # Toy environment simulation (e.g., GridWorld)
    agent = QLearningAgent(n_states=5, n_actions=2)
    
    print("--- Q-Learning Update ---")
    print(f"Initial Q-value for State 0, Action 1: {agent.q_table[0, 1]:.1f}")
    
    # Simulate an agent taking action 1 in state 0, receiving reward 10, and moving to state 1
    agent.update(state=0, action=1, reward=10.0, next_state=1, done=False)
    
    print(f"Updated Q-value for State 0, Action 1: {agent.q_table[0, 1]:.1f}")
