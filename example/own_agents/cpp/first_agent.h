#pragma once

/**
 * TODO: comment
 * 
 * Author: Caitlin
 */

#include "HFO.hpp"
#include "reinforcement_learning.h"


class FirstAgent {
public:
  void start();

private:
  // Create the HFO environment
  hfo::HFOEnvironment HFO_;
  // Server Connection Options. See printouts from bin/HFO.
  hfo::feature_set_t FEATURES_ = hfo::HIGH_LEVEL_FEATURE_SET;
  std::string CONFIG_DIR_ = "bin/teams/base/config/formations-dt";
  int PORT_ = 6000;
  std::string SERVER_ADDRESS_ = "localhost";
  std::string TEAM_NAME_ = "base_left";
  bool GOALIE_ = false;

  // Amount states and actions
  int AMOUNT_STATES = 16; // TODO: depends on amount teammates and opponents
  int AMOUNT_ACTIONS = 2; // 7
  RL rl_;

  void performAction(unsigned int action);
  float getDistanceToBall(std::vector<float> states);
};



