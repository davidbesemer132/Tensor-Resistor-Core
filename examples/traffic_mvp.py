# Traffic Control MVP

This script simulates a 3-lane traffic control system using modal logic and karmic debt tracking.

## Features
- Simulates traffic flow across three lanes.
- Implements modal logic for decision making (e.g., green light, red light).
- Tracks karmic debt based on traffic violations.

## Code Structure
The main simulation loop runs through a fixed number of iterations, checking the state of each lane and updating the traffic signals accordingly. Violations are recorded, affecting the karmic debt of the driver.

### Code Snippet
Here's a brief overview of the core functionality:

```python
class TrafficLight:
    def __init__(self, lane):
        self.lane = lane
        self.state = "red"

    def change_light(self, new_state):
        self.state = new_state

class TrafficController:
    def __init__(self):
        self.lights = [TrafficLight(i) for i in range(3)]

    def simulate(self, iterations):
        for i in range(iterations):
            # Logic for changing the lights
            pass

controller = TrafficController()
controller.simulate(100)
```

## Experimentation Instructions
Run the script in your Python environment. Modify the parameters in `simulate()` method to see how traffic changes with different configurations.
