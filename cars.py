import math
import numpy as np
class car:
    def __init__(self, color, x, y, destination):
        self.color = color
        self.rotation = 0
        self.velocity = 0
        self.acceleration = 0
        self.max_velocity = 20
        self.max_acceleration = 5
        self.crashed = False
        self.success = False
        self.x = x  
        self.y = y
        self.destination = destination
        self.space_occupied = []
        for i in range(6):
            for j in range(12):
                self.space_occupied.append([self.x + i-3,self.y + j-6])

    def rotate(self, theta):
       try: 
        self.rotation += theta
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[c,-s],[s,c]])
        new_vec = []
        for i, coord in enumerate(self.space_occupied):
            c = np.array(coord)
            c[0] -= self.x 
            c[1] -= self.y
            c = np.matmul(c,r)
            c[0] += self.x 
            c[1] += self.y
            c[0] = int(c[0])
            c[1] = int(c[1])
            new_vec.append(c.tolist())
        self.space_occupied = new_vec
       except:
        print(self.rotation) 
        print(c)

    def forward(self, amount):
       try: 
        d_x = int(math.cos(self.rotation) * amount)
        d_y = int(math.sin(self.rotation) * amount)
    

        for i, coords in enumerate(self.space_occupied):
            self.space_occupied[i][0] += d_y
            self.space_occupied[i][1] += d_x
        self.x += d_x
        self.y += d_y
       except:
        print(self.rotation) 
    def move(self, amount):
        if self.crashed:
            return
        adj_amount = -1 * math.cos(3.14 * 2 * amount)
        if amount  < -0.25:
            rotate_amount = (abs(amount) - 0.33)/0.67
        elif amount > 0.25:
            rotate_amount  =-1 * (abs(amount) - 0.33)/0.67
        else:
            rotate_amount = 0
        self.forward(self.velocity * 3)
        self.rotate(0.5 * math.pi * (rotate_amount - rotate_amount*self.velocity/20))

        if self.acceleration > self.max_acceleration:
            self.acceleration = self.max_acceleration
        elif self.acceleration < -self.max_acceleration:
            self.acceleration = -self.max_acceleration

        self.velocity += self.acceleration

        if self.velocity > self.max_velocity:
            self.velocity = self.max_velocity
        elif self.velocity < -self.max_velocity:
            self.velocity = -self.max_velocity

        self.acceleration = ( adj_amount) * 5
  
    def render(self):
        return self.space_occupied, self.color




