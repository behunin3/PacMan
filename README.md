obs, reward, terminated, truncated, info = env.step(action)
reward is a float. Per iteration** (**double check)
terminated and truncated are booleans
info is a dictionary: {'lives': 3, 'episode_frame_number': 4, 'frame_number': 4}