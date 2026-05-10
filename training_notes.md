# Training Notes for PCC-RL Project

## Project Overview
- **Name**: PCC-RL (Performance-oriented Congestion Control with Reinforcement Learning)
- **Purpose**: Train reinforcement learning models for internet congestion control
- **Based on**: Paper "A Reinforcement Learning Perspective on Internet Congestion Control" (ICML 2019)

## Key Components
- **Environment**: Custom Gym environment 'PccNs-v0' for network simulation
- **Training Script**: `src/gym/stable_solve.py` using PPO1 from stable-baselines
- **Network Simulation**: `src/gym/network_sim.py`
- **Online Testing**: Integration with PCC-Uspace for real-world testing

## Dependencies
- Python packages: gym<0.26, stable-baselines (old version ~0.7), tensorflow==1.15.0
- Related repo: github.com/PCCProject/PCC-Uspace for real-world deployment

## Training Setup
- Location: `src/gym/`
- Command: `python stable_solve.py`
- Default architecture: 32,16 (configurable via --arch)
- Gamma: 0.99 (configurable via --gamma)

## Current Issues and Status
- TensorFlow 1.15.0 is not available on PyPI for Python 3.7 (deprecated)
- Attempting to install Miniconda for environment management
- Once Miniconda is installed, will create Python 3.7 environment with required packages
- Alternative: Modify script to use stable-baselines3 and TensorFlow 2.x (requires code changes)

## Important Notes During Training
- Training may take significant time depending on hardware
- Monitor for convergence of the RL model
- Check for any errors in network simulation
- Model saves periodically (check code for save paths)

## Next Steps
- After training, test the model using the online components
- Refer to `src/gym/online/README.md` for testing instructions