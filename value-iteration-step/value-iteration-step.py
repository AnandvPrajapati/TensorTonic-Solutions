def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    new_values = []

    # Iterate through each state
    for s in range(len(values)):
        action_values = []

        # Iterate through each action available in state s
        for a in range(len(transitions[s])):
            expected_future_value = 0.0

            # Sum over all possible next states
            for s_next in range(len(values)):
                expected_future_value += (
                    transitions[s][a][s_next] * values[s_next]
                )

            # Bellman update for Q(s, a)
            q_value = rewards[s][a] + gamma * expected_future_value
            action_values.append(q_value)

        # Take the maximum action value
        new_values.append(float(max(action_values)))

    return new_values