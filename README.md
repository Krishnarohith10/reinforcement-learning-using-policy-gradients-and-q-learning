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

cart_pole_v0_using_policy_gradients.py - Training our reinforcement learning agent using policy gradients technique. Where our PG algorithm optimize the policy parameters following gradients towards higher rewards.  

ms_pacman_v0_deep_q_learning.py - Training our reinforcment learning agent using q learning technique. Where we create two similar models, actor_model which is used to predict the state-action values. And other critic_model to predict the future state_action values and used for training actor_model. In the end, we use trained actor_model to predict action possibily the action with maximum state value. 

Note: Running this python file, requires more memory, which is definitely bigger than 32GB. I have 32GB memory and I trained for only till 1+ lakh Iteration. Reinforcement Learning needs more memory, whereas Deep Learning needs more GPU memory.
