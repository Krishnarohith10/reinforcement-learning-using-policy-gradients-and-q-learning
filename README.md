# reinforcement-learning-using-policy-gradients-and-q-learning

Reinforcement Learning (RL) is one of the most exciting fields in Machine Learning today. In Reinforcement Learning, a software agent makes observations and takes action in an environment, and in return it gets rewards. It comes under special category, we only know, Supervised and Unsupervised but it is other category. In this Reinforcement Learning, the agent learns only throught rewards and evaluates an action based on receiving rewards.  

There are mainly two man methods in Reinforcement Learning:  
    1) Policy Gradients (PG)  
    2) Deep Q Networks (DQN)  

Policy Gradients: It is the method used exactly as gradient descent in other machine learning algorithms. But here it is called gradients ascent. The policy gradients (PG) algorithm optimize the parameters of policy by following gradients towards higher rewards.  

Deep Q Networks: On the other hand, Deep Q Networks follows Q-Learning algorithm. An algorithm similar to the Q-value iteration algorithm by Richard Bellman.  

For more details please go through Reinforcement Learning in book "Hands-On Machine Learning with Scikit-Learn and Tensorflow" by Aurelien Geron.  

In this repository we used the task of Balancing Pole on a moving Cart.  

Libraries used:  
    gym - OpenAI gym is a toolkit that provides a wide variety of simulated environments (Atari games, board games, 2D and 3D physical simulations, and so on), so you can train agents, compare them, or develop new RL algorithms.  
    tensorflow - library for machine learning  

policy_gradients.py - Training our reinforcement learning agent using policy gradients technique. Where our PG algorithm optimize the policy parameters following gradients towards higher rewards.  
