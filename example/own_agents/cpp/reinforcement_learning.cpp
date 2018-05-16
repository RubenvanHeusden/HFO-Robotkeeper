/**
 * TODO: comment
 * 
 * Author: Caitlin
 */

#include <iostream>
#include <math.h>
#include "reinforcement_learning.h"

RL::RL() : AMOUNT_STATES(0), AMOUNT_ACTIONS(0) {
  std::cout << "Initialising reinforcement learning..." << std::endl;
}

RL::RL(int amount_states, int amount_actions) : AMOUNT_STATES(tiny_dnn::serial_size_t(amount_states)),
                                                AMOUNT_ACTIONS(tiny_dnn::serial_size_t(amount_actions)) {
  std::cout << "Initialising reinforcement learning..." << std::endl;
  initNetwork();
}

bool RL::setStateVector(const std::vector<float> &states) {
  states_ = new_states_;
  q_values_ = new_q_values_;
  action_ = new_action_;

  RL::Position robot_position = RL::Position(states[0], states[1], states[2]);
  RL::Position ball_position = RL::Position(states[3], states[4], 0);

  RL::Position relative_position_ball = globalToRelative(robot_position, ball_position);

  // TODO: int van maken
  std::vector<float> discretised_states(AMOUNT_STATES);
  int state = discretisePosition(relative_position_ball, states[6] > 0) - 1;
  discretised_states[state] = 1;

  if (states_.size() > 0 && states_[state] == 1) return false; // Return if state not changed

  new_states_ = discretised_states;
  std::cout << "Ball velocity: " << states[5] << std::endl;
  if (after_kick_action_ != 255 && ++amount_states_after_kick_ > 2 && states[5] == -1 ) resetAfterKick();

  std::cout << "State: " << state << std::endl;
  if (states_.size() == 0) {
    std::cout << "Begin trial" << std::endl;
    std::cout << "First state: " << state << std::endl;
    states_ = new_states_;
  }
  return true;
}

unsigned int RL::selectAction() {
  std::cout << "Selecting new action..." << std::endl;
  new_action_ = 255;
  printFloatVector(new_states_, "state for selecting action");
  // Predict qValues
  new_q_values_ = nn_.predict(new_states_);
  printTinyDnnVector(new_q_values_, "Q-values");
  if (SELECTION_METHOD == "e-greedy") {
    if (((rand() % 10) / 10.f) < 0.3) {
      std::cout << "Selecting random action." << std::endl;
      new_action_ = rand() % AMOUNT_ACTIONS;
    }
  }
  if (SELECTION_METHOD == "greedy" || (SELECTION_METHOD == "e-greedy" && new_action_ == 255)) {
    new_action_ = static_cast<unsigned int>(std::distance(new_q_values_.begin(), std::max_element(new_q_values_.begin(), new_q_values_.end())));
  }

  if (SELECTION_METHOD == "boltzmann") {
    double max_q_value = double(*std::max_element(new_q_values_.begin(), new_q_values_.end()));
    double sum_probabilities = 0;
    std::vector<double> probabilities(AMOUNT_ACTIONS);
    for (int i = 0; i < AMOUNT_ACTIONS; ++i) {
      probabilities[i] = exp((new_q_values_[i] - max_q_value) / TAU);
      sum_probabilities += probabilities[i];
//      std::cout << "qvalue: " << new_q_values_[i] << std::endl;
      // std::cout << "e: " << probabilities[i] << std::endl;
    }
    // std::cout << "sum_probabilities: " << sum_probabilities << std::endl;

    for (int i = 0; i < AMOUNT_ACTIONS; ++i) {
      probabilities[i] /= sum_probabilities;
//      std::cout << "prob " << i << ": " << probabilities[i] << std::endl;
    }

    // Get random action from probability distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(probabilities.begin(), probabilities.end());
    new_action_ = static_cast<unsigned int>(d(gen));
  }

  if (q_values_.size() == 0) {
    std::cout << "First action in trial" << std::endl;
    std::cout << "First action: " << new_action_ << std::endl;
    q_values_ = new_q_values_;
    action_ = new_action_;
  }
  return new_action_;
}


void RL::updateQValues(float reward) {
  std::cout << "Updating Q-values..." << std::endl;

  std::vector<tiny_dnn::vec_t> input;
  tiny_dnn::vec_t tiny_dnn_state_vector(states_.begin(), states_.end());
//  printTinyDnnVector(tiny_dnn_state_vector, "Tiny_dnn_state_vector that will be updated");
  input.push_back(tiny_dnn_state_vector);

  // Update Q values
  std::cout << "Update q value with: " << (*std::max_element(new_q_values_.begin(), new_q_values_.end()) - q_values_[action_]) << " and reward: " << reward << std::endl;
  q_values_[action_] += ALPHA*(reward + GAMMA *(*std::max_element(new_q_values_.begin(), new_q_values_.end()) - q_values_[action_]));

//  printTinyDnnVector(q_values_, "q-values that is updated");

  // Train network
  tiny_dnn::gradient_descent opt;
  opt.alpha = 0.00025;
  std::vector<tiny_dnn::vec_t> output;
  output.push_back(q_values_);
  nn_.fit<tiny_dnn::mse>(opt, input, output, BATCH_SIZE, EPOCHS);
}


void RL::updateQValuesAfterTrial(float reward) {
  std::cout << "Updating Q-values after trial..." << std::endl;

  // If scored and after kick is set, use after kick states, q_values and action
  // TODO: ^

  // TODO: extra update weghalen

  if (after_kick_action_ != 255) {
    std::cout << "Changed to after kick" << std::endl;
    new_states_ = new_after_kick_states_;
    new_q_values_ = new_after_kick_q_values_;
    new_action_ = new_after_kick_action_;
    states_ = after_kick_states_;
    q_values_ = after_kick_q_values_;
    action_ = after_kick_action_;
    nn_ = after_kick_nn_;
  }

  // TEST
  std::vector<float> test_start(AMOUNT_STATES);
  test_start[10] = 1;
  tiny_dnn::vec_t test_input_start(test_start.begin(), test_start.end());

  tiny_dnn::vec_t test_predict_start = nn_.predict(test_input_start);
  printTinyDnnVector(test_predict_start, "Q-values test start predict before");
  // END TEST

  std::vector<tiny_dnn::vec_t> input;
  tiny_dnn::vec_t tiny_dnn_state_vector(new_states_.begin(), new_states_.end());
//  printTinyDnnVector(tiny_dnn_state_vector, "Tiny_dnn_state_vector that will be updated");
  input.push_back(tiny_dnn_state_vector);
//  tiny_dnn::vec_t tiny_dnn_state_vector2(states_.begin(), states_.end());
//  printTinyDnnVector(tiny_dnn_state_vector2, "Tiny_dnn_state_vector2 that will be updated");
//  input.push_back(tiny_dnn_state_vector2);


  printTinyDnnVector(new_q_values_, "new q-values before updated");
//  printTinyDnnVector(q_values_, "q-values before updated");
  // Update Q values
  new_q_values_[new_action_] += reward;
//  q_values_[action_] += reward;

  printTinyDnnVector(new_q_values_, "new q-values that are updated");
//  printTinyDnnVector(q_values_, "q-values that are updated");

  // Train network
  tiny_dnn::gradient_descent opt;
  opt.alpha = 0.00025;
  std::vector<tiny_dnn::vec_t> output;
  output.push_back(new_q_values_);
//  output.push_back(q_values_);
  nn_.fit<tiny_dnn::mse>(opt, input, output, BATCH_SIZE, EPOCHS);


  // TEST
  tiny_dnn::vec_t test_predict = nn_.predict(new_states_);
  printTinyDnnVector(test_predict, "Q-values test predict");

  tiny_dnn::vec_t test_predict_start_after = nn_.predict(test_input_start);
  printTinyDnnVector(test_predict_start_after, "Q-values test start predict");
  // END TEST

//  std::cout << "**************************" << std::endl;
//  tiny_dnn::vec_t* weights = nn_[0]->weights()[0];
//  for (float w : *weights) {
//    std::cout << w << std::endl;
//  }
//  std::cout << "**************************" << std::endl;
//  states_ = new_states_;
//  q_values_ = new_q_values_;
//  action_ = new_action_;
  states_.clear();
//  new_states_.clear();
  q_values_.clear();
//  new_q_values_.clear();
  action_ = 255;
//  new_action_ = 255;
  resetAfterKick();
}


int RL::discretisePosition(RL::Position position, bool is_kickable) {
  int angle_bin = int(round((position.theta / ( 2.0 * M_PI)) * 8)) + 4;
  if (angle_bin == 0) angle_bin = 8; // set zero bin to 8

  // Compute distance
  float distance_to_ball = float(pow(pow(position.x, 2) + pow(position.y, 2), 0.5));
  // TODO: iets moois bedenken dat niet zo random is?
  int distance_bin = AMOUNT_STATES / 8;
  if (is_kickable)
    distance_bin = 1;
  else if (distance_to_ball < 0.2 && AMOUNT_STATES > 2 * 8)
    distance_bin = 2;
  else if (distance_to_ball < 0.3 && AMOUNT_STATES > 3 * 8)
    distance_bin = 3;
  else if (distance_to_ball < 0.4  && AMOUNT_STATES > 4 * 8)
    distance_bin = 4;

  return angle_bin + ((distance_bin - 1) * 8);
}

RL::Position RL::globalToRelative(RL::Position robot_position, RL::Position object_position) {

  float temp_x = object_position.x - robot_position.x;
  float temp_y = object_position.y - robot_position.y;

  RL::Position relative_position;

  relative_position.x = temp_x * (float) cos(robot_position.theta) + temp_y * (float) sin(robot_position.theta);
  relative_position.y = -temp_x * (float) sin(robot_position.theta) + temp_y * (float) cos(robot_position.theta);

  relative_position.theta = float(std::atan2(double(relative_position.y), double(relative_position.x)));

  return relative_position;
}

void RL::initNetwork() {
  nn_ << tiny_dnn::fc<tiny_dnn::activation::relu>(AMOUNT_STATES, AMOUNT_STATES)
//     << tiny_dnn::fc<tiny_dnn::activation::relu>(AMOUNT_STATES, AMOUNT_STATES)
//     << tiny_dnn::fc<tiny_dnn::activation::relu>(AMOUNT_STATES, AMOUNT_STATES)
     << tiny_dnn::fc<tiny_dnn::activation::relu>(AMOUNT_STATES, AMOUNT_STATES)
     << tiny_dnn::fc<tiny_dnn::activation::identity>(AMOUNT_STATES, AMOUNT_ACTIONS);

  nn_.weight_init(tiny_dnn::weight_init::xavier(1.0));
  nn_.bias_init(tiny_dnn::weight_init::constant(0.0));
  nn_.init_weight();

  for (int i = 0; i < int(nn_.depth()); i++) {
    std::cout << "#layer:" << i << "\n";
    std::cout << "layer type:" << nn_[i]->layer_type() << "\n";
    std::cout << "input:" << nn_[i]->in_size() << "(" << nn_[i]->in_shape() << ")\n";
    std::cout << "output:" << nn_[i]->out_size() << "(" << nn_[i]->out_shape() << ")\n";
    std::cout << "weights size:" << nn_[i]->weights()[0]->size() << "\n";
    std::cout << std::endl;
  }

  std::cout << std::endl;
}

void RL::printFloatVector(std::vector<float> vector, std::string name) {
  std::cout << "Printing " << name << std::endl;
  for (float i : vector) {
      std::cout << i << std::endl;
  }
}

void RL::printTinyDnnVector(tiny_dnn::vec_t vector, std::string name) {
  std::cout << "Printing " << name << std::endl;
  for (float i : vector) {
    std::cout << i << std::endl;
  }
}

void RL::setAfterKick() {
  if (after_kick_action_ != 255) return; // Set only if not already set
//  printFloatVector(new_states_, "after kick states that are set");
  new_after_kick_states_ = new_states_;
  new_after_kick_q_values_ = new_q_values_;
  new_after_kick_action_ = new_action_;
  after_kick_states_ = states_;
  after_kick_q_values_ = q_values_;
  after_kick_action_ = action_;
  after_kick_nn_ = nn_;
  ++amount_states_after_kick_;
}

void RL::resetAfterKick() {
  std::cout << "Reset after kick" << std::endl;
  after_kick_states_.clear();
  after_kick_q_values_.clear();
  after_kick_action_ = 255;
  new_after_kick_states_.clear();
  new_after_kick_q_values_.clear();
  new_after_kick_action_ = 255;
  after_kick_nn_ = nn_;
  amount_states_after_kick_ = 0;
}

std::vector<float> RL::getStateVector() {
  return new_states_;
}
