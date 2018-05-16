#pragma once

/**
 * TODO: comment
 * 
 * Author: Caitlin
 */


#include <vector.h>
#include <common.hpp>
#include "tiny_dnn/tiny_dnn.h"


class RL {
  struct Position {
    float x;
    float y;
    float theta;
    Position() {}
    Position(float x, float y, float theta) : x(x), y(y), theta(theta) {}
  };

public:
  RL();
  RL(int amount_states, int amount_actions);
  bool setStateVector(const std::vector<float>& states);
  std::vector<float> getStateVector();
  unsigned int selectAction();
  void updateQValues(float reward);
  void updateQValuesAfterTrial(float reward);
  void setAfterKick();
  void resetAfterKick();

private:
  tiny_dnn::network<tiny_dnn::sequential> nn_;
  std::vector<float> states_;
  std::vector<float> new_states_;
  tiny_dnn::vec_t q_values_;
  tiny_dnn::vec_t new_q_values_;
  unsigned int action_;
  unsigned int new_action_;

  unsigned int amount_states_after_kick_ = 0;
  std::vector<float> after_kick_states_;
  std::vector<float> new_after_kick_states_;
  tiny_dnn::vec_t after_kick_q_values_;
  tiny_dnn::vec_t new_after_kick_q_values_;
  unsigned int after_kick_action_ = 255;
  unsigned int new_after_kick_action_ = 255;
  tiny_dnn::network<tiny_dnn::sequential> after_kick_nn_;

  tiny_dnn::serial_size_t AMOUNT_STATES; // TODO: depends on amount teammates and opponents
  tiny_dnn::serial_size_t AMOUNT_ACTIONS;
  float GAMMA = 0.99f;
  float ALPHA = 1;
  size_t BATCH_SIZE = 1;
  int EPOCHS = 1;
  float TAU = 0.8f;
  std::string SELECTION_METHOD = "e-greedy";

  void initNetwork();
  RL::Position globalToRelative(RL::Position robot_position, RL::Position object_position);
  int discretisePosition(RL::Position position, bool is_kickable);

  void printFloatVector(std::vector<float> vector, std::string name);
  void printTinyDnnVector(tiny_dnn::vec_t vector, std::string name);
};



