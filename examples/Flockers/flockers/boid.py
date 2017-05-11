import numpy as np

from mesa import Agent


class Boid(Agent):
    '''
    A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents.
        - Separation: avoiding getting too close to any other agent.
        - Alignment: try to fly in the same direction as the neighbors.

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and heading (a unit vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    '''
    cohere_factor = .025
    separate_factor = .25
    match_factor = .04
    def __init__(self, unique_id, model, pos, speed, velocity, vision, separation):
        '''
        Create a new Boid flocker agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby Boids.
            separation: Minimum distance to maintain from other Boids.
        '''
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed
        self.velocity = velocity
        self.vision = vision
        self.separation = separation

    def cohere(self, neighbors):
        '''
        Return the vector toward the center of mass of the local neighbors.
        '''
        cohere = np.zeros(2)
        for neighbor in neighbors:
            cohere += self.model.space.get_heading(self.pos, neighbor.pos) / len(neighbors)
        return cohere

    def separate(self, neighbors):
        '''
        Return a vector away from any neighbors closer than separation dist.
        '''
        close = lambda n:self.model.space.get_distance(n.pos, self.pos) < self.separation
        separation_vector = np.zeros(2)
        for neighbor in filter(close, neighbors):
            separation_vector -= self.model.space.get_heading(self.pos, neighbor.pos)
        return separation_vector

    def match_heading(self, neighbors):
        '''
        Return a vector of the neighbors' average heading.
        '''
        neighbor_count = len(neighbors)
        match_vector = np.zeros(2)
        for neighbor in neighbors:
            match_vector += neighbor.velocity / neighbor_count
        return match_vector

    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''

        neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        self.velocity += (self.cohere(neighbors) * self.cohere_factor
                        + self.separate(neighbors) * self.separate_factor
                        + self.match_heading(neighbors) * self.match_factor) / 2
        self.velocity /= np.linalg.norm(self.velocity)
        new_pos = self.pos + self.velocity * self.speed
        self.model.space.move_agent(self, new_pos)
