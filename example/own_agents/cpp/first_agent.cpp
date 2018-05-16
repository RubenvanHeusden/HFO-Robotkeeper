/**
 * TODO: comment
 * 
 * Author: Caitlin
 */
#include <zconf.h>
#include "first_agent.h"

void FirstAgent::start() {
  // Connect to the server and request high-level feature set.
  HFO_.connectToServer(FEATURES_, CONFIG_DIR_, PORT_, SERVER_ADDRESS_,
                      TEAM_NAME_, GOALIE_);
  hfo::status_t status = hfo::IN_GAME;

  rl_ = RL(AMOUNT_STATES, AMOUNT_ACTIONS);
  int episode;
  for (episode = 0; status != hfo::SERVER_DOWN; ++episode) {
    status = hfo::IN_GAME;
    std::vector<float> states = HFO_.getState();
    float distance = getDistanceToBall(states);
    rl_.setStateVector(states);
    unsigned int action = rl_.selectAction();
    float reward = 0;
    while (status == hfo::IN_GAME) {
      performAction(action);
      states = HFO_.getState();
      if (!rl_.setStateVector(states)) {
        status = HFO_.step();
        usleep(1000);
        continue;
      }
      action = rl_.selectAction();
      float new_distance = getDistanceToBall(states);
      if (new_distance < distance) {
        reward = 1 - new_distance;
      } else {
        reward = -1 * new_distance;
      }
      distance = new_distance;
      std::cout << "Reward: " << reward << std::endl;
      rl_.updateQValues(reward);

      // Advance the environment and receive current game status
      status = HFO_.step();
      usleep(1000);
    }
    std::cout << "Last action: " << action << std::endl;
    if (status == hfo::GOAL) {
      reward = 20.f;
    } else {
      reward = -5.f;
    }

    rl_.updateQValuesAfterTrial(reward);

    // Check what the outcome of the episode was
    std::cout << "Episode " << episode << " ended with status: "
         << StatusToString(status) << std::endl;
  }
  std::cout << "Episode " << episode << " ended with status: "
            << StatusToString(status) << std::endl;
  HFO_.act(hfo::QUIT);
}

void FirstAgent::performAction(unsigned int action) {
  switch (action) {
    case 0:
      // Kick left
      HFO_.act(hfo::KICK, 100.0, -20.0);
      rl_.setAfterKick();
//      std::cout << "Performed kick left" << std::endl;
      break;
    case 1:
      // Kick right
      // TODO: remove walk here
      HFO_.act(hfo::DASH, 100.0, 0.0);
//      HFO_.act(hfo::KICK, 100.0, 20.0);
//      rl_.setAfterKick();
//      std::cout << "Performed kick right" << std::endl;
      break;
    case 2:
      // Kick forward
      HFO_.act(hfo::KICK, 100.0, 0.0);
      rl_.setAfterKick();
//      std::cout << "Performed kick forward" << std::endl;
      break;
    case 3:
      // Walk forward
      HFO_.act(hfo::DASH, 100.0, 0.0);
//      std::cout << "Performed walk forward" << std::endl;
      break;
    case 4:
      // Walk backward
      HFO_.act(hfo::DASH, -100.0, 0.0);
//      std::cout << "Performed walk backward" << std::endl;
      break;
    case 5:
      // Walk left
      HFO_.act(hfo::DASH, 100.0, -90.0);
//      std::cout << "Performed walk left" << std::endl;
      break;
    case 6:
      // Walk right
      HFO_.act(hfo::DASH, 100.0, 90.0);
//      std::cout << "Performed walk right" << std::endl;
      break;
    default:
      // Stand
      HFO_.act(hfo::DASH, 0.0, 0.0);
//      std::cout << "Performed stand" << std::endl;
      break;
  }

}

float FirstAgent::getDistanceToBall(std::vector<float> states) {
  return float(std::pow(std::pow((states[0] - states[3]), 2) + std::pow((states[1] - states[4]), 2), 0.5));
}

int main() {
  FirstAgent agent;
  agent.start();
};
