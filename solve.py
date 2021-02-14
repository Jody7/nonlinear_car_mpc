from pyomo.environ import *
from pyomo.dae import *
import pyomo.contrib.fbbt.interval as interval

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import math

solver = SolverFactory('ipopt')

epsilon = 1e-2

mass = 0.041
inertia_z = 27.8e-6
l_f = 0.04 # wheel base length
l_r = 0.07

pajecka_mult = 0.8
# tire model constants
B_f = 3.38
C_f = 1.2 * 1.2 * 1.2
D_f = 0.192
B_r = 3.38*pajecka_mult 
C_r = 1.2*pajecka_mult
D_r = 0.192

# electric drivetrain and friction constants
C_r0 = 0.0518
C_r2 = 0.00035
C_m1 = 0.287
C_m2 = 0.054

obstacle_circles = []

def solve():
	mpc_time = 4
	mpc_samples = 120

	m = ConcreteModel()
	m.t = ContinuousSet(bounds=(0, mpc_time))

	m.D = Var(m.t) # throttle
	m.delta = Var(m.t) # steering angle
	m.D_dot = DerivativeVar(m.D)
	m.delta_dot = DerivativeVar(m.delta)

	m.constr_D_dot_upper = Constraint(m.t, rule=lambda m, t: (m.D_dot[t]) <= 3)
	m.constr_D_dot_lower = Constraint(m.t, rule=lambda m, t: (m.D_dot[t]) >= -3)
	m.constr_delta_dot_upper = Constraint(m.t, rule=lambda m, t: m.delta_dot[t] <= math.radians(180))
	m.constr_delta_dot_lower = Constraint(m.t, rule=lambda m, t: m.delta_dot[t] >= -math.radians(180))

	m.obstacle_penalty = Var(m.t)
	m.x = Var(m.t)
	m.y = Var(m.t)
	m.phi = Var(m.t) # heading
	m.v_x = Var(m.t)
	m.v_y = Var(m.t)
	m.omega = DerivativeVar(m.phi) # angular vel

	m.x_dot = DerivativeVar(m.x)
	m.y_dot = DerivativeVar(m.y)
	m.v_x_dot = DerivativeVar(m.v_x)
	m.v_y_dot = DerivativeVar(m.v_y)
	m.omega_dot = DerivativeVar(m.omega)

	m.F_y_f = Var(m.t)
	m.F_y_r = Var(m.t)
	m.F_x_r = Var(m.t)
	m.alpha_f = Var(m.t)
	m.alpha_r = Var(m.t)

	m.pc = ConstraintList()
	#m.pc.add(m.D[0]==0)
	#m.pc.add(m.delta[0]==0)
	m.pc.add(m.x[0]==0)
	m.pc.add(m.y[0]==0)
	m.pc.add(m.phi[0]==0)
	m.pc.add(m.v_x[0]==0)
	m.pc.add(m.v_y[0]==0)
	m.pc.add(m.omega[0]==0)
	#m.pc.add(m.x_dot[0]==0)
	#m.pc.add(m.y_dot[0]==0)
	#m.pc.add(m.v_x_dot[0]==0)
	#m.pc.add(m.v_y_dot[0]==0)
	m.pc.add(m.omega_dot[0]==0)
	#m.pc.add(m.F_y_f[0]==0)
	#m.pc.add(m.F_y_r[0]==0)
	#m.pc.add(m.F_x_r[0]==0)

	#m.constr_input_1 = Constraint(m.t, rule=lambda m, t: m.D[t] == 0.1)
	#m.constr_input_2 = Constraint(m.t, rule=lambda m, t: m.delta[t] == math.radians(15))

	obstacle_circles.append([1,1,0.5])
	obstacle_circles.append([2.25,1.25,0.7])
	obstacle_circles.append([0.9, -0.2, 0.5])
	obstacle_circles.append([0.6,0.2,0.25])
	obstacle_circles.append([0,1.5,1])
	#obstacle_circles.append([2.4,1,0.25])

	m.constr_1 = Constraint(m.t, rule=lambda m, t: m.v_x[t] >= -0.1)
	m.constr_2 = Constraint(m.t, rule=lambda m, t: m.D[t] <= 1)
	m.constr_3 = Constraint(m.t, rule=lambda m, t: m.D[t] >= -0.5)
	m.constr_4 = Constraint(m.t, rule=lambda m, t: m.delta[t] <= math.radians(45))
	m.constr_5 = Constraint(m.t, rule=lambda m, t: m.delta[t] >= -math.radians(45))

	# kinematic constraints
	m.x_dot_ode = Constraint(m.t, rule=lambda m, t: 
		m.x_dot[t] == m.v_x[t] * cos(m.phi[t]) - m.v_y[t] * sin(m.phi[t])
	)
	m.y_dot_ode = Constraint(m.t, rule=lambda m, t: 
		m.y_dot[t] == m.v_x[t] * sin(m.phi[t]) + m.v_y[t] * cos(m.phi[t])
	)
	# dynamics constraints
	m.v_x_dot_ode = Constraint(m.t, rule=lambda m, t: 
		m.v_x_dot[t] == 1/mass * (m.F_x_r[t] - m.F_y_f[t]*sin(m.delta[t]) + mass*m.v_y[t]*m.omega[t])
	)
	m.v_y_dot_ode = Constraint(m.t, rule=lambda m, t: 
		m.v_y_dot[t] == 1/mass * (m.F_y_r[t] + m.F_y_f[t]*cos(m.delta[t]) - mass*m.v_x[t]*m.omega[t])
	)
	m.omega_dot_ode = Constraint(m.t, rule=lambda m, t: 
		m.omega_dot[t] == 1/inertia_z * (m.F_y_f[t]*l_f*cos(m.delta[t]) - m.F_y_r[t]*l_r)
	)
	# tire / drivetrain dynamics constraints
	m.F_y_f_ode = Constraint(m.t, rule=lambda m, t: 
		m.F_y_f[t] == D_f*sin(C_f*atan(B_f*m.alpha_f[t]))
	)
	m.F_y_r_ode = Constraint(m.t, rule=lambda m, t: 
		m.F_y_r[t] == D_r*sin(C_r*atan(B_r*m.alpha_r[t]))
	)
	m.F_x_r_ode = Constraint(m.t, rule=lambda m, t: 
		m.F_x_r[t] == (C_m1 - C_m2*m.v_x[t])*m.D[t] - C_r0 - C_r2*pow(m.v_x[t], 2)
	)
	m.alpha_f_ode = Constraint(m.t, rule=lambda m, t: 
		m.alpha_f[t] == -atan((m.omega[t]*l_f + m.v_y[t]) / (m.v_x[t] + epsilon)) + m.delta[t]
	)
	m.alpha_r_ode = Constraint(m.t, rule=lambda m, t: 
		m.alpha_r[t] == atan((m.omega[t]*l_r - m.v_y[t]) / (m.v_x[t] + epsilon))
	)

	def cost_function_integral(m, t):
		return 100*(m.x[t] - 2.5)**2 + 100*(m.y[t] - 2.5)**2 #+ 999999*(m.obstacle_penalty[t])**2

	m.integral = Integral(m.t, wrt=m.t, rule=cost_function_integral)
	m.obj = Objective(expr=m.integral +
		+ 0.1*(m.x[m.t[-1]]-2.5)**2 + 0.1*(m.y[m.t[-1]]-2.5)**2)

	TransformationFactory('dae.collocation').apply_to(m, wrt=m.t, nfe=mpc_samples, ncp=1, scheme='LAGRANGE-RADAU')
	#TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=mpc_samples)
	solver.options['max_iter'] = 9000

	for i in range(len(obstacle_circles)):
		def circle_radius_expr(m, t):
			return sqrt((m.x[t]-obstacle_circles[i][0])**2 + (m.y[t]-obstacle_circles[i][1])**2) >= (obstacle_circles[i][2] - 0*m.obstacle_penalty[t])
		setattr(m, 'obstacle_circles_constraint' + str(i), Constraint(m.t, rule=circle_radius_expr))

	solver.solve(m, tee=False)

	plot_list = {}
	plot_params = ['delta', 'phi', 'omega_dot', 'v_x', 'v_y', 'D', 'F_y_f', 'F_y_r', 'F_x_r']
	for element in plot_params:
		uniq_mul = 1
		if element == 'delta' or element == 'phi' or element == 'omega':
			uniq_mul = math.degrees(1)
		elif element == 'D':
			uniq_mul = 10
		elif element == 'F_y_f' or element == 'F_y_r' or element == 'F_x_r':
			uniq_mul = 100
		plot_list[element] = np.array([getattr(m, element)[t]()*uniq_mul for t in m.t]).tolist()

	t_array = np.array([t for t in m.t]).tolist()
	x_array = np.array([m.x[t]() for t in m.t]).tolist()
	y_array = np.array([m.y[t]() for t in m.t]).tolist()

	steering_pos_x = []
	steering_pos_y = []
	front_wheel_pos_x = []
	front_wheel_pos_y = []
	back_wheel_pos_x = []
	back_wheel_pos_y = []

	for t in m.t:
		steering_based_multiplier = 0.5
		wheel_based_multiplier = 1.25
		phi_multiplier = 1
		phi_offset = math.radians(0)
		front_wheel_pos_x.append(m.x[t]() + wheel_based_multiplier*l_f*cos(phi_offset+phi_multiplier*m.phi[t]()))
		front_wheel_pos_y.append(m.y[t]() + wheel_based_multiplier*l_f*sin(phi_offset+phi_multiplier*m.phi[t]()))
		back_wheel_pos_x.append(m.x[t]() + -wheel_based_multiplier*l_r*cos(phi_offset+phi_multiplier*m.phi[t]()))
		back_wheel_pos_y.append(m.y[t]() + -wheel_based_multiplier*l_r*sin(phi_offset+phi_multiplier*m.phi[t]()))
		#back_wheel_pos_x.append(m.x[t]())
		#back_wheel_pos_y.append(m.y[t]())

		steering_pos_x.append(front_wheel_pos_x[-1] + steering_based_multiplier*l_f*cos(m.delta[t]()+phi_offset+phi_multiplier*m.phi[t]()))
		steering_pos_y.append(front_wheel_pos_y[-1] + steering_based_multiplier*l_f*sin(m.delta[t]()+phi_offset+phi_multiplier*m.phi[t]()))

	for key in plot_list:
		plt.plot(t_array,plot_list[key],'',label=key,marker='o')
		plt.legend()

	fig = plt.figure()
	ax = plt.axes(xlim=(-0.5, 3.25), ylim=(-0.5, 3.25))

	circle_goal = plt.Circle((2.5, 2.5), 0.1, color='blue')
	ax.add_patch(circle_goal)
	
	for i in range(len(obstacle_circles)):
		ax.add_patch(plt.Circle((obstacle_circles[i][0], obstacle_circles[i][1]), obstacle_circles[i][2]-0.01, color='red'))

	lines = []

	def animate(anim_i):
		#if len(lines) > 0:
			#for i in range(len(lines)):
				#lines[i].remove()
			#lines.clear()
		if anim_i == 0:
			lines.clear()

		i = anim_i - 1
		line, = ax.plot([front_wheel_pos_x[i], back_wheel_pos_x[i]], [front_wheel_pos_y[i], back_wheel_pos_y[i]], linewidth=3.0)
		line_steer, = ax.plot([steering_pos_x[i], front_wheel_pos_x[i]], [steering_pos_y[i], front_wheel_pos_y[i]], linewidth=3.0, color='black')
		lines.insert(i, line)
		lines.insert(i, line_steer)

		return tuple(lines)

	anim = animation.FuncAnimation(fig, animate, frames=len(front_wheel_pos_y), interval=100, blit=True, repeat=True)
	f = r"out.gif" 
	writergif = animation.PillowWriter(fps=10) 
	anim.save(f, writer=writergif)

	plt.show()

solve()