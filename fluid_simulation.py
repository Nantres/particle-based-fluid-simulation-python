import math
import numpy as np
import taichi as ti
import cv2

ti.init(arch=ti.gpu)

# variables u can change
activate_gui = False
preset = 0

# don't touch these variables
gui_res = (512, 288)
display_res = (1080, 1820)
gui_to_display_multiplier = display_res[0] / gui_res[1]
bg_color = 0xFFFFFF
dimensions = 2
particle_spacing = 3
time_of_button_pressed = 0
time_delta = 1.0 / 120.0
epsilon = 1e-5
interaction_radius = 70
interaction_strength = 350
blank_image = np.zeros((display_res[0], display_res[1], 3), np.uint8)

if preset == 0:
    number_of_particles = 1200
    particle_radius = 3
if preset == 1:
    number_of_particles = 50
    particle_radius = 20
boundary_min = particle_radius
boundary_max = ti.Vector([gui_res[0] - epsilon, gui_res[1]]) - particle_radius


# ti variables
pixels = ti.Vector.field(3, dtype=ti.f32, shape=gui_res)
positions = ti.Vector.field(dimensions, dtype=ti.f32, shape=number_of_particles)
velocities = ti.Vector.field(dimensions, dtype=ti.f32, shape=number_of_particles)
term_n = ti.field(dtype=ti.i32, shape=())
gravity_vector = ti.Vector([0.0, -700.0])
densities = ti.field(dtype=ti.f32, shape=number_of_particles)
mouse_position = ti.field(dtype=ti.f32, shape=2)
particle_cell_keys =  ti.field(dtype=ti.i32, shape=number_of_particles)
particle_index_sorted_by_cell =  ti.field(dtype=ti.i32, shape=number_of_particles)
start_indices = ti.field(dtype=ti.i32, shape=number_of_particles)

# ti widgets
gui = ti.GUI("PBF2D", gui_res)
target_density_slider = gui.slider('Target Density', 0, 10, step=0.1)
target_density_slider.value = 6
smoothing_radius_slider = gui.slider('Smoothing Radius', 0.1, 50, step=1)
smoothing_radius_slider.value = 10
pressure_multiplier_slider = gui.slider('Pressure Multiplier', 1, 2000, step=10)
pressure_multiplier_slider.value = 1000
gravity_slider = gui.slider('Gravity', 0, 1000, step=10)
gravity_slider.value = 700


@ti.func
def confine_position_to_boundary(i):
    position = positions[i]
    for dimension in ti.static(range(dimensions)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if position[dimension] <= boundary_min:
            position[dimension] = boundary_min + epsilon * ti.random()
            velocities[i][dimension] *= -0.9
        elif boundary_max[dimension] <= position[dimension]:
            position[dimension] = boundary_max[dimension] - epsilon * ti.random()
            velocities[i][dimension] *= -0.9
    return position

@ti.func
def smoothing_kernel(distance, smoothing_radius) -> ti.f32:
    influence = 0.0
    if distance >= smoothing_radius:
        influence = 0.0
    else:
        volume = math.pi * smoothing_radius ** 4 / 6
        influence = (smoothing_radius - distance) * (smoothing_radius - distance) / volume
    return influence

@ti.func
def smoothing_kernel_derivative(distance, smoothing_radius) -> ti.f32:
    slope = 0.0
    if distance >= smoothing_radius:
        slope = 0.0
    else:
        scale = 12 / (math.pi * (smoothing_radius ** 4))
        slope = (distance - smoothing_radius) * scale
    return slope

@ti.func
def calculate_density(sample_point, smoothing_radius):
    density = 0.0
    for i in range(number_of_particles):
        distance = (positions[i] - sample_point).norm()
        influence = smoothing_kernel(distance, smoothing_radius)
        density += influence
    return density


@ti.func
def convert_density_to_pressure(density, target_density, pressure_multiplier) -> ti.f32:
    density_error = density - target_density
    pressure = density_error * pressure_multiplier
    return pressure

@ti.func
def calculate_shared_pressure(density_A, density_B, target_density, pressure_multiplier):
    pressure_A = convert_density_to_pressure(density_A, target_density, pressure_multiplier)
    pressure_B = convert_density_to_pressure(density_B, target_density, pressure_multiplier)
    return (pressure_A + pressure_B) / 2

@ti.func
def calculate_interaction_force(input_pos, radius, strength, particle_index):
    interaction_force = ti.Vector([0.0, 0.0])
    offset = input_pos - positions[particle_index]
    offset_squared = offset.dot(offset)
    if (offset_squared < radius * radius):
        distance = ti.sqrt(offset_squared)
        direction_to_input_point = ti.Vector([0.0, 0.0]) if distance <= epsilon else offset / distance
        centreT = 1 - distance / radius
        interaction_force += (direction_to_input_point * strength - velocities[particle_index]) * centreT
        interaction_acceleration = interaction_force / densities[particle_index]
        velocities[particle_index] += interaction_acceleration * time_delta   


@ti.kernel
def update_start_indices():
    for i in positions:
        key = particle_cell_keys[i]
        key_prev = -1 if i == 0 else particle_cell_keys[i-1]
        if key != key_prev:
            start_indices[key] = i


@ti.kernel
def calculate_pressure_force(smoothing_radius: ti.f32, target_density: ti.f32, pressure_multiplier: ti.f32, mouse_input: ti.i32):
    for sample_index in positions:
        sample_point = positions[sample_index]
        pressure_force = ti.Vector([0.0, 0.0])
        for i in range(number_of_particles):
            if sample_index == i:
                continue
            vector_distance = sample_point - positions[i]
            vector_magnitude = vector_distance.norm()    # distance between particle and sample point
            if vector_magnitude > smoothing_radius:
                continue
            direction = vector_distance / vector_magnitude if vector_magnitude != 0 else (0, 1) # direction vector
            slope = smoothing_kernel_derivative(vector_magnitude, smoothing_radius)
            density = densities[i]
            shared_pressure = calculate_shared_pressure(density, densities[sample_index], target_density, pressure_multiplier)
            pressure_force += shared_pressure * slope * direction / density

        pressure_acceleration = pressure_force / densities[sample_index]
        velocities[sample_index] += pressure_acceleration * time_delta
        if mouse_input == 1:
            calculate_interaction_force(ti.Vector([mouse_position[0], mouse_position[1]]), interaction_radius, interaction_strength, sample_index)
        elif mouse_input == 2:
            calculate_interaction_force(ti.Vector([mouse_position[0], mouse_position[1]]), interaction_radius, -interaction_strength, sample_index)
    
@ti.kernel
def gravity_and_density(smoothing_radius: ti.f32, gravity_value: ti.f32):
    gravity_vector = ti.Vector([0.0, -gravity_value])
    for i in positions:
        velocities[i] += gravity_vector * time_delta
        densities[i] = calculate_density(positions[i], smoothing_radius)
        update_position(i)

@ti.kernel
def initialize():
    a = 30  # Initial margin (padding)
    d = 2 * particle_radius + particle_spacing  # Spacing between particles
    cols = (gui_res[0] - 2 * a) // d  # Compute number of columns dynamically
    for n in positions:
        row = n // cols  # Compute row index
        col = n % cols    # Compute column index
        x = a + col * d + ti.random() * 2
        y = a + row * d + ti.random() * 2
        positions[n] = ti.Vector([x, y])

@ti.func
def update_position(i: ti.i32):
    positions[i] += velocities[i] * time_delta
    positions[i] = confine_position_to_boundary(i)


def simulation_step(mouse_input):
    gravity_and_density(smoothing_radius_slider.value, gravity_slider.value)
    calculate_pressure_force(smoothing_radius_slider.value, target_density_slider.value, pressure_multiplier_slider.value, mouse_input)
    # print(particle_cell_keys)
    # print(start_indices)
    # print(particle_index_sorted_by_cell)


def render(gui):
    gui.clear(bg_color)
    gui.circles(positions.to_numpy()/gui_res, radius=particle_radius, color=0x000000)
    gui.show()

def draw_circles(img, multiplier):
    pos_np = positions.to_numpy()
    for pos in pos_np:
        x, y = int(pos[0]*multiplier), display_res[0] - int(pos[1]*multiplier)
        cv2.circle(img, (x,y), 7, color=(255,0,0), thickness=-1)


initialize()
play = False
mouse_input = 0
show_gui = True
step = False
while gui.running and not gui.get_event(gui.ESCAPE):
    gui.get_event()
    if gui.is_pressed(gui.DOWN) and play:
        play = False
        print("paused")
    elif gui.is_pressed(gui.UP) and not play:
        play = True
        print("playing")
    elif gui.is_pressed(gui.LEFT):
        show_gui = False
    elif gui.is_pressed(gui.RIGHT):
        show_gui = True
    elif gui.is_pressed(gui.LMB):
        mouse_input = 1
    elif gui.is_pressed(gui.RMB):
        mouse_input = 2
    else:
        mouse_input = 0
    if mouse_input != 0:
        mouse_position[0], mouse_position[1] = gui.get_cursor_pos()
        mouse_position[0] *= gui_res[0]
        mouse_position[1] *= gui_res[1]
    if play:
        simulation_step(mouse_input)
    elif step:
        simulation_step(mouse_input)
        step = False
    if show_gui:
        render(gui)
    display = np.zeros((display_res[0], display_res[1], 3), np.uint8)
    draw_circles(display, gui_to_display_multiplier)
    cv2.namedWindow('display',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('display', display)
    cv2.waitKey(1)

    
