# Quoridor Game Machine Learning Model Specification

## Inputs:
- **verwalls**:
  - *Description*: N x N matrix representing vertical walls.
  - *Type*: Boolean matrix where `True` indicates the presence of a wall.
  - *Constraints*: Last row and column are filled with `0`s due to game rules limiting the placement of walls.

- **horwalls**:
  - *Description*: N x N matrix representing horizontal walls.
  - *Type*: Boolean matrix where `True` indicates the presence of a wall.
  - *Constraints*: Last row and column are filled with `0`s due to game rules limiting the placement of walls.

- **playerPos**:
  - *Description*: N x N matrix indicating player positions.
  - *Type*: Integer matrix where `1` represents Player 1, `-1` represents Player 2, and `0` for empty spaces.
  - *Purpose*: Optimized for CNN locality to capture spatial relationships.

- **player1Walls**:
  - *Description*: Integer indicating the number of walls remaining for Player 1.
  - *Type*: Integer.

- **player2Walls**:
  - *Description*: Integer indicating the number of walls remaining for Player 2.
  - *Type*: Integer.

## Outputs:
- **Model 1: Win Probability Prediction**:
  - *Description*: Predicts the probability (ranging from 0 to 1) of Player 1 winning the game based on the current board state.
  - *Type*: Single float value representing the predicted probability. Used to evaluate positons

- **Model 2: Action Selection Prediction**:
  - *Description*: Predicts the likelihood of success for a set of game actions from the current board state.
  - *Type*: Predictive model outputting probabilities or scores for possible game actions. Used to prune game tree and select actions
  - *Thoughts*: There might not exist a optimal move set (or it is not known) so this might be better suited for use in RL of some sort.
