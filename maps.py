import numpy as np, cv2
import random
import cv2 as cv

def create_map(roads, destinations):
    road_map = np.zeros((1000,1000), np.float32)
    prev_road = False
    road_locations_vertical = []
    road_locations_horizontal = []
    colors_used = []
    destination_colors = [[255, 0, 0],[241, 255, 0],  [123, 0, 255], [0, 0, 255]]
    for i in range(roads):
        location = random.randint(100, 900)
        if not prev_road:
            while (location in road_locations_vertical) or (location + 25 in road_locations_vertical) or (location - 5 in road_locations_vertical) :
                location = random.randint(100, 900)
            for j in range(len(road_map)):
                for k in range(20):
                    place = location+k
                    road_locations_vertical.append(place)
                    road_map[j][place] = 1
            prev_road = True

        else:
            while (location in road_locations_horizontal) or (location + 25 in road_locations_horizontal) or (location - 5 in road_locations_horizontal) :
                location = random.randint(100, 900)

            for j in range(len(road_map[0])):
                for k in range(20):
                    place = location+k
                    road_locations_horizontal.append(place)
                    road_map[place][j] = 1
            prev_road = False

    road_map1 = cv2.cvtColor(road_map, cv2.COLOR_GRAY2BGR)
    destination_vec = []
    for i in range(destinations):
        direction = random.randint(0,1)
        if direction == 0:
            place_x = random.choice(road_locations_vertical)
            while not (place_x + 10 in road_locations_vertical and place_x < 990):
                place_x = random.choice(road_locations_vertical)
            place_y = random.randint(0,990)
            color_index = random.randint(0, len(destination_colors) -1)
            color = destination_colors[color_index]
            colors_used.append(color)
            destination_vec.append((place_x+5, place_y+5))

            destination_colors.pop(color_index)

            for j in range(10):
                for k in range(10):
                    road_map1[place_y + j][place_x + k] = color

        else:
            place_y = random.choice(road_locations_horizontal)
            while not (place_y + 10 in road_locations_horizontal and place_y < 990):
                place_y = random.choice(road_locations_horizontal)            
            place_x = random.randint(0,990)
            color_index = random.randint(0, len(destination_colors) -1)
            color = destination_colors[color_index]
            colors_used.append(color)
            destination_vec.append((place_x+5, place_y+5))
            destination_colors.pop(color_index)

            for j in range(10):
                for k in range(10):
                    road_map1[place_y + j][place_x + k] = color
    return road_map1, colors_used,destination_vec, road_locations_horizontal, road_locations_vertical
 