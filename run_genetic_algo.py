from maps import create_map
from cars import car
import random
import numpy as np
import cv2 as cv
import numpy as np
from PIL import Image
from cnn import *
import time
from matplotlib import pyplot as plt

def display_image(c):
    #c1 = np.transpose(c, (2,0,1))
    plt.imshow(c, interpolation='nearest')
    plt.show()
def show_car_only(car):
    z = np.zeros((100,100, 3), np.float32)
    for coord in car.space_occupied:
        y = int(int(coord[1]) - car.y + 10)
        x = int(int(coord[0]) - car.x + 10)
        z[y][x] == np.array(car.color)
    img = Image.fromarray(z, 'RGB')
    img.show()
def shuffle(a, b, c):
    t = list(zip(a,b,c))
    random.shuffle(t)
    a,b,c = zip(*t)
    return list(a), list(b), list(c)

def render_cars(cars, map1):
    for car in cars:
        for coord in car.space_occupied:
           try: 
            
            map1[int(coord[1])][int(coord[0])] = car.color
           except:
            pass 
        for i in range(3):
            for j in range(3):
               try: 
                k = i-1
                l = j-1
                map1[car.destination[1]-k][car.destination[0]-l] = car.color
               except:
                pass 
    return map1
def render_cars_color(car, map1):
    row1 = np.repeat(np.array(car.color), len(map1))
    col1 = np.repeat(np.array(car.color), len(map1)+2)
    row1.reshape(1,-1,3)

    map1 = np.array(map1)
 

    map1 = np.vstack((map1, row1.reshape(1,-1,3)))
    map1 = np.vstack((map1, row1.reshape(1,-1,3)))
    
    map1 = np.hstack((map1, col1.reshape(-1,1,3)))
    map1 = np.hstack((map1, col1.reshape(-1,1,3)))

    return map1
def calculate_hits(cars, maps):
    hits = 0
    success = 0
    for i, car in enumerate(cars):
        for coord in car.space_occupied:
           if not car.crashed: 
            try:
                c1 = int(coord[1])
                c2 = int(coord[0])
                if c1 < 0 or c2 < 0:
                    cars[i].crashed = True
                    hits += 1
                    continue

                p = maps[int(coord[1])][int(coord[0])]
                if not (p[0] == 1.0 and p[1] == 1.0 and p[2] == 1.0) :
                    if (p[0] == car.color[0] and p[1] == car.color[1] and p[2] == car.color[2]):
                        success += 1
                        cars[i].success = True
                    else:
                        hits += 1
                    cars[i].crashed = True
            except IndexError:             
                hits += 1
                cars[i].crashed = True
                
    return hits,success, cars

def calculate_success(cars):
    success = 0
    for car in cars:
        if car.destination in car.space_occupied and not car.crashed:
            success += 1
            car.crashed = True
    return success, cars    

def calculate_loss(cars, r_h, r_v,  i):
    distance = 0
    hits = len(cars)
    for car in cars:
        if not car.crashed:
            d = calculate_distance([car.x,car.y], list(car.destination), r_h, r_v, 0)
            distance += d
        if car.success:
            hits -= 1

    loss = ((distance +5) *1.5 + hits*3000) * (i+10)/100    
    return loss
def calculate_distance(l1, l2, r_h, r_v, distance):

    if l1[0] in r_v and l1[1] in r_h:
        distance += abs(l2[1] - l1[1])
        distance += abs(l2[0] - l1[0]) 
        return distance
    else:
        distance += abs(l2[1] - l1[1])*2
        distance += abs(l2[0] - l1[0]) 
        return distance
def all_finished(cars):
    for car in cars:
        if not car.crashed:
            return False
    return True
def train_l_net(net, x1c, x2c, yc):
    print("TRAINING LOSS NET")
 
    for i in range(10):
        x1, x2, y = shuffle(x1c.copy(), x2c.copy(),yc.copy())
        
        for i, data in enumerate(x1):
            
            data = np.array(data)
            x = x2[i].detach().clone().to(net.device)
            output = net.forward(data, x)
            y1 = torch.tensor([y[i]]).to(net.device)
            loss = net.criterion(output, y1.view(1,-1).detach().clone())
            print(loss)
            loss.backward()
            net.optimizer.step()
    return net



def training_loop(ga_instance, solution, sol_idx):
    global model, map1,r_h, r_v, destinations, colors_used
    print("m")
    cars = create_cars(colors_used, r_v, r_h, destinations)

    i = 100
    finished = False

    while i > 0 and not finished:
        m1 = render_cars(cars,map1.copy())
        l = 0

        for car in cars:
            
            c =  np.array(car.color)
            i_tensor = torch.tensor([[i]]).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            color = torch.from_numpy(c).float().to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            x = torch.cat(( color.view(1,-1), i_tensor.view(1,-1)) ,1)
            m = torch.cat(torch.from_numpy(m1).view(1,-1) , x, 1)
            predictions = pygad.torchga.predict(model=model,
                                        solution=solution,
                                        data=m)           

            output_i = predictions[0].item()

            car.move(output_i)
        p = calculate_hits(cars, map1.copy())
        finished = all_finished(cars)
        if finished:
            print("finished loop" + str(i))
        if i % 10 == 0:
            print("step: " + str(i))
            print("loss: " + str(l.item()/len(cars)))
        i -= 1
    
    hits, success, cars = calculate_hits(cars, map1.copy())
    loss = calculate_loss(cars, r_h, r_v,  i)
    print("ACTUALL LOSS: " + str(loss))
   
    return  1000/(loss+0.1)

def create_cars(colors_used, road_locations_vertical, road_locations_horizontal, destinations):
    cars = []
    for i, colors in enumerate(colors_used):

        place = random.randint(0,1)
        mod_x = 0
        mod_y = 0
        if place == 0:
            place_x = random.choice(road_locations_vertical)
            while not (place_x + 14 in road_locations_vertical):
                place_x = random.choice(road_locations_vertical)
            place_y = random.randint(0,1000)
            mod_x = 7
        else:
            place_y = random.choice(road_locations_horizontal)
            while not (place_y + 14 in road_locations_horizontal):
                place_y = random.choice(road_locations_horizontal)
            place_x = random.randint(0,1000)
            mod_y = 7

        car1 = car(colors, place_x + mod_x, place_y + mod_y, destinations[i])
        cars.append(car1)
    return cars


def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")
    
import pygad.torchga as torchga
import pygad
map1, colors_used,destinations, r_h, r_v = create_map(10, 2)
model = CNN(1)
torch_ga = torchga.TorchGA(model=model,
                           num_solutions=20)
num_generations = 100 # Number of generations.
num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights
print("c0")

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=training_loop,
                       on_generation=on_generation)
print("c1")
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")        
    



def run(training_loops, num_cars, num_roads):
    
    p_net = CNN( 1)
    lx1 = []
    lx2 = []
    lx3 = []

    cars = create_cars(colors_used, road_locations_vertical, road_locations_horizontal, destinations)
    ly = []
    mx = render_cars(cars, map1.copy())
    loss_array = []
    avg_loss = 1
    start = True
    i = 0
    while True:
        i += 1
        cars = create_cars(colors_used, road_locations_vertical, road_locations_horizontal, destinations)

        p_net,  loss  = training_loop(p_net, map1, cars, road_locations_horizontal, road_locations_vertical, avg_loss)
        loss_array.append(loss)
        avg_loss = loss/(sum(loss_array)/len(loss_array))
        print("Loop: " + str(i))
        if i%1 == 0 and not i ==0:
            x = np.array(range(len(loss_array)))
            y = np.array(loss_array)
            plt.title("Line graph")
            plt.plot(x, y, color="red")
            plt.savefig('results.png')
    

       
        
