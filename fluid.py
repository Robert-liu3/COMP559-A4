## Robert Liu 260981372

import moderngl_window as mglw
import moderngl as mgl
import numpy as np
import glm as glm

class Source:
	'''A source of temperature (heating or cooling) in the fluid.'''
	def __init__(self, x, y, strength):
		self.x = x
		self.y = y
		self.strength = strength
		if strength > 0: self.colour = (1, 0, 0, 1)
		else: self.colour = (0, 0, 1, 1)

class HelloWorld(mglw.WindowConfig):
	'''
	Stable Fluid Simulation

	Note that array indexing is row column, so the first index is the y index and the second index is the x index.
	Note that there is an extra layer of cells outside the domain, so integer indexing within the actual domain is from 1 to nx, 1 to ny.
	Note that the velocities are stored at the centers of the cells.
	The spatial limits of the domain are given by xl xh yl yh.
	'''
	
	nx = 32 # TODO: Change the resolution to different sizes in testing!
	ny = 16

	pix_per_cell = 1024/(nx+4) # Set the cell size to get a nice window size 
	win_width = pix_per_cell*(nx+4)
	win_height = pix_per_cell*(ny+4)
	gl_version = (3, 3)
	title = "Stable Fluid Simulation - YOUR NAME AND STUDENT ID HERE"
	window_size = (win_width, win_height)
	aspect_ratio = win_width / win_height
	resizable = True
	resource_dir = 'data'

	color_samples = 5 # sample temperature colours at a higher resolution than the grid for better visualisation

	yl = -1.0
	yh =  1.0
	dy = 2.0/ny
	dx = dy
	xl = -dx*nx/2.0 
	xh =  dx*nx/2.0 

	draw_grid_enabled = True
	draw_grid_lines_enabled = True
	draw_velocity_enabled = True
	velocity_scale = 0.1
	step_request = False
	reset_request = False
	running = False
	num_particles = 5000

	iterations = 32	    # number of Gauss-Seidel solver iterations
	
	dt = 0.1			# time step (control with up and down arrows)
	viscosity = 0.00001 # viscosity of fluid (diffuision of velocities) (control with 1 and 2 keys)
	kappa = 0.00001	    # thermal diffusivity (control with 3 and 4 keys)
	beta = 0.05         # buoyancy force control (control with 5 and 6 keys)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)  
		self.ctx.enable(mgl.BLEND)
		self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
		self.prog = self.ctx.program( vertex_shader = open('glsl/vert.glsl').read(), fragment_shader = open('glsl/frag.glsl').read() )
		self.prog2 = self.ctx.program( vertex_shader = open('glsl/constantColourVert.glsl').read(), fragment_shader = open('glsl/constantColourFrag.glsl').read() )
		# transformation for drawing the fluid grid
		self.MVP = glm.ortho( self.xl-self.dx*2, self.xh+self.dx*2, self.yl-self.dy*2, self.yh+self.dy*2, -1, 1)
		self.prog['MVP'].write( self.MVP )
		self.prog2['MVP'].write( self.MVP )
		self.star_vao = self.load_scene("star.obj").root_nodes[0].mesh.vao.instance(self.prog2) 
		self.reset()
		self.setup_grid_lines()
		self.setup_velocity_lines()
		self.setup_boundary_box()
		self.setup_color_mesh()
		self.setup_particles()
		self.sources = []

	def reset(self):
		self.curr_v = np.zeros((self.ny+2, self.nx+2, 2)) # velocities
		self.next_v = np.zeros((self.ny+2, self.nx+2, 2)) # velocities
		self.curr_tp = np.zeros((self.ny+2, self.nx+2)) # temperature
		self.next_tp = np.zeros((self.ny+2, self.nx+2)) # temperature

	def xy_to_ij(self, x, y):
		'''Converts x,y position to i,j indices and bilinear interpolation coefficients a,b.'''
		# Accounting for padding one expects at xl, yl, i=0, j=0 with a=b=0.5
		j = np.floor((x-self.xl)/self.dx+0.5).astype(int)
		i = np.floor((y-self.yl)/self.dy+0.5).astype(int)
		a = (x-self.xl)/self.dx+0.5 - j
		b = (y-self.yl)/self.dy+0.5 - i
		return i, j, a, b

	def setup_color_mesh(self):
		''' Sets up drawing of a coloured mesh on the domain to show temperatures '''
		# Note the mesh resolution is higher than the grid resolution for better visualisation
		x,y = np.meshgrid(np.linspace(self.xl,self.xh,self.nx*self.color_samples), np.linspace(self.yl,self.yh,self.ny*self.color_samples))
		self.cg_i,self.cg_j,self.cg_a,self.cg_b = self.xy_to_ij(x,y) # store color grid (cg) indices and interpolation coefficients
		vertices = np.column_stack((x.flatten(), y.flatten())).astype('f4')
		# Triangle strip would be better, but just making *ALL* the triangles is easier
		ind = np.reshape( range(x.size), (self.ny*self.color_samples,self.nx*self.color_samples))
		j,i = np.meshgrid( range(self.nx*self.color_samples-1), range(self.ny*self.color_samples-1))
		indices1 = np.column_stack((ind[i, j].flatten(), ind[i, j + 1].flatten(), ind[i + 1, j].flatten()))
		indices2 = np.column_stack((ind[i, j + 1].flatten(), ind[i + 1, j + 1].flatten(), ind[i + 1, j].flatten()))
		indices = np.concatenate( (indices1, indices2), axis=0 )
		vbo_pos = self.ctx.buffer(vertices.astype('f4').tobytes())
		self.vbo_col = self.ctx.buffer(np.zeros((x.size,4)).astype('f4').tobytes())	
		ibo = self.ctx.buffer(indices.astype("i4").tobytes())	
		self.grid_vao = self.ctx.vertex_array( self.prog,
			[ (vbo_pos, '2f', 'in_position'), (self.vbo_col, '4f', 'in_colour') ], index_buffer=ibo, mode=mgl.TRIANGLES )

	def draw_color_mesh(self):
		''' Updates the temperature mesh colours and draws it. '''
		i,j,a,b = self.cg_i,self.cg_j,self.cg_a,self.cg_b 
		## bilinear interpolation
		value = (1-a)*(1-b)*self.curr_tp[i,j] + (1-a)*b*self.curr_tp[i+1,j] + a*(1-b)*self.curr_tp[i,j+1] + a*b*self.curr_tp[i+1,j+1]
		colours = np.zeros((i.size,4), dtype='f4')
		colours[:,0] = np.clip(value.flatten(), 0, 1)
		colours[:,2] = np.clip(-value.flatten(), 0, 1)
		colours[:,3] = 1
		self.vbo_col.write(colours.astype("f4").tobytes())
		self.grid_vao.render()

	def setup_velocity_lines(self):
		'''Create a vertex array with repeated mesh grid values for drawing velocity lines.'''
		x = np.linspace( self.xl-0.5*self.dx, self.xh+0.5*self.dx, self.nx+2)
		y = np.linspace( self.yl-0.5*self.dy, self.yh+0.5*self.dy, self.ny+2)
		xg, yg = np.meshgrid(x,y)		
		self.vertices = np.zeros((xg.size*2,2), dtype='f4')  # *2 for center and center plus velocity
		self.vertices[0::2,0] = xg.flatten()
		self.vertices[0::2,1] = yg.flatten()
		self.vertices[1::2,0] = xg.flatten() 
		self.vertices[1::2,1] = yg.flatten() 
		self.vbo = self.ctx.buffer( self.vertices.astype("f4").tobytes(),dynamic=True )
		self.lines_vao = self.ctx.vertex_array(self.prog2, [(self.vbo, '2f', 'in_position')], mode=mgl.LINES)
	
	def draw_velocity_lines(self):
		'''Draw velocities after updating the line segment enpoints in the dynamic draw vbo of positions'''
		self.vertices[1::2,0] = self.vertices[0::2,0] + self.velocity_scale*self.curr_v[:,:,0].flatten()
		self.vertices[1::2,1] = self.vertices[0::2,1] + self.velocity_scale*self.curr_v[:,:,1].flatten()
		self.vbo.write(self.vertices.astype("f4").tobytes())
		self.lines_vao.program['colour'] = (0, 1, 0, 0.5)
		self.lines_vao.render()

	def reset_particles(self):
		'''Randomly distribute particles in the domain.'''
		x = np.random.uniform(self.xl, self.xh, self.num_particles)
		y = np.random.uniform(self.yl, self.yh, self.num_particles)
		self.particles = np.column_stack((x, y))

	def setup_particles(self):
		self.reset_particles()
		self.particle_vbo = self.ctx.buffer(self.particles.astype('f4').tobytes())
		self.particle_vao = self.ctx.vertex_array(self.prog2, [(self.particle_vbo, '2f', 'in_position')], mode=mgl.POINTS)
	
	def draw_particles(self):
		self.particle_vbo.write(self.particles.astype('f4').tobytes())
		self.particle_vao.program['colour'] = (1, 1, 1, 1)
		self.particle_vao.render()

	def setup_grid_lines(self):
		'''Creates a vao for drawing the grid lines (note there is one layer of cells outside the domain)'''
		# have nx*ny cells, but padding by one extra on all sides if not using MAC grid
		verticesy = np.zeros(((self.ny+3)*2,2), dtype='f4') # set vertices to be horizontal lines
		verticesy[0::2,0] = self.xl - self.dx
		verticesy[0::2,1] = np.linspace(self.yl-self.dy, self.yh+self.dy, self.ny+3)
		verticesy[1::2,0] = self.xh + self.dx
		verticesy[1::2,1] = np.linspace(self.yl-self.dy, self.yh+self.dy, self.ny+3)
		verticesx = np.zeros(((self.nx+3)*2,2), dtype='f4') # set vertices to be vertical lines
		verticesx[0::2,0] = np.linspace(self.xl-self.dx, self.xh+self.dx, self.nx+3)
		verticesx[0::2,1] = self.yl - self.dy
		verticesx[1::2,0] = np.linspace(self.xl-self.dx, self.xh+self.dx, self.nx+3)
		verticesx[1::2,1] = self.yh + self.dy
		vertices = np.concatenate((verticesx, verticesy), axis=0) # concatenate the two arrays
		vbo_pos = self.ctx.buffer(vertices.astype('f4').tobytes())
		self.grid_lines_vao = self.ctx.vertex_array( self.prog2, [ (vbo_pos, '2f', 'in_position')], mode=mgl.LINES)

	def setup_boundary_box(self):
		'''Creates a vao for drawing a box showing the domain (note there is one layer of cells outside the domain)'''
		vertices = np.array([[self.xl, self.yl],[self.xl, self.yh],[self.xh, self.yh],[self.xh, self.yl]])
		vbo_pos = self.ctx.buffer(vertices.astype('f4').tobytes())
		self.domain_box_vao = self.ctx.vertex_array( self.prog2, [ (vbo_pos, '2f', 'in_position')], mode=mgl.LINE_LOOP)

	def key_event(self, key, action, modifiers):
		if action == self.wnd.keys.ACTION_PRESS:
			if key == self.wnd.keys.V: self.draw_velocity_enabled = not self.draw_velocity_enabled	
			if key == self.wnd.keys.G: self.draw_grid_lines_enabled = not self.draw_grid_lines_enabled	
			if key == self.wnd.keys.C: self.draw_grid_enabled = not self.draw_grid_enabled	
			if key == self.wnd.keys.P: self.reset_particles()
			if key == self.wnd.keys.R: self.reset_request = True
			if key == self.wnd.keys.S: self.step_request = True	
			if key == self.wnd.keys.SPACE: self.running = not self.running	
			if key == self.wnd.keys.ESCAPE: self.close()
			if key == self.wnd.keys.LEFT: self.velocity_scale *= 0.5
			if key == self.wnd.keys.RIGHT: self.velocity_scale *= 2.0
			if key == self.wnd.keys.COMMA: 
				self.iterations = np.clip(self.iterations/2, 2, 256).astype(int)
				print("Iterations:", self.iterations)
			if key == self.wnd.keys.PERIOD: 
				self.iterations = np.clip(self.iterations*2, 2, 256).astype(int)
				print("Iterations:", self.iterations)
			if key == self.wnd.keys.UP: 
				self.dt *= 2.0
				print("dt:", self.dt)
			if key == self.wnd.keys.DOWN: 
				self.dt *= 0.5
				print("dt:", self.dt)
			if key == self.wnd.keys.NUMBER_1: 
				self.viscosity *= 0.5
				print("Viscosity:", self.viscosity)
			if key == self.wnd.keys.NUMBER_2: 
				self.viscosity *= 2.0
				print("Viscosity:", self.viscosity)
			if key == self.wnd.keys.NUMBER_3:
				self.kappa *= 0.5
				print("Thermal diffusivity:", self.kappa)
			if key == self.wnd.keys.NUMBER_4:
				self.kappa *= 2.0
				print("Thermal diffusivity:", self.kappa)
			if key == self.wnd.keys.NUMBER_5:
				self.beta *= 0.5
				print("Buoyancy force control:", self.beta)
			if key == self.wnd.keys.NUMBER_6:
				self.beta *= 2.0
				print("Buoyancy force control:", self.beta)
			if key == self.wnd.keys.DELETE:
				self.sources = []
				
	def mouse_press_event(self, x, y, button):
		xx, yy = self.mouse_to_xy(x, y)
		if (xx < self.xl or xx > self.xh or yy < self.yl or yy > self.yh): return
		if button == self.wnd.mouse.left:
			self.sources.append(Source(xx, yy, -1))
		if button == self.wnd.mouse.right:
			self.sources.append(Source(xx, yy, 1))

	def mouse_to_xy(self, x, y):
		'''Converts mouse coordinates to x,y position in the domain.'''
		(w,h) = self.wnd.size
		self.pix_per_cell = np.min((w/(self.nx+4), h/(self.ny+4)))
		xx = (x-w/2)/self.pix_per_cell*self.dx
		yy = (h/2-y)/self.pix_per_cell*self.dy
		return xx, yy

	def mouse_drag_event(self, mouse_x, mouse_y, mouse_dx, mouse_dy):
		''' Applies a force to the velocity grid. '''
		xx, yy = self.mouse_to_xy(mouse_x, mouse_y)
		if (xx < self.xl or xx > self.xh or yy < self.yl or yy > self.yh): return
		i,j,a,b = self.xy_to_ij(xx,yy)
		dv = 0.1 * np.array( [mouse_dx, -mouse_dy] )
		if ( i >= 0 and i <= self.ny and j >= 0 and j <= self.nx):
			self.curr_v[i,j,:] += (1-a)*(1-b)*dv
			self.curr_v[i+1,j,:] += (1-a)*b*dv
			self.curr_v[i,j+1,:] += a*(1-b)*dv
			self.curr_v[i+1,j+1,:] += a*b*dv
		self.curr_v[:,:,0] = self.set_boundary(self.curr_v[:,:,0], "horizontal")
		self.curr_v[:,:,1] = self.set_boundary(self.curr_v[:,:,1], "vertical")

	# Code below this point is the fluid simulation code  

	def diffuse(self, q0, kappa, dt, boundary_type):
		'''Solve the diffusion equation using the implicit method.'''
		# print("diffusion is being called")
		# q = q0.copy()
		# a=dt*kappa*self.iterations**2
		# for k in range(20):
		# 	for i in range(self.iterations):
		# 		for j in range(self.iterations):
		# 			q[1:-1, 1:-1] = (q0[1:-1, 1:-1] + a*(q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]))/(1+4*a)
		# 			q = self.set_boundary(q, boundary_type)
		x = np.arange(self.nx+1)
		y = np.arange(self.ny+1)
		X, Y = np.meshgrid(x, y)


		N = 1/self.dx
		a = dt * kappa * self.nx * self.ny
		q = q0.copy()



		for _ in range(self.iterations):
			q[1:self.ny+1, 1:self.nx+1] = (q0[1:self.ny+1, 1:self.nx+1] + a * (q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:])) / (1 + 4 * a)
			self.set_boundary(q, boundary_type)
			# for i in range(1, self.ny+1):
			# 	for j in range(1, self.nx+1):
			# 		q[i, j] = (q0[i, j] + a * (q[i-1, j] + q[i+1, j] + q[i, j-1] + q[i, j+1])) / (1 + 4 * a)
			# self.set_boundary(q, boundary_type)
		return q
	
			# TODO: STEP 3: Complete this function
		# return q0 placeholder, but return the correct result as a np.array

	def set_boundary(self, q, boundary_type):
		# print("this is q", q)
		# print("this is boundary_type", boundary_type)
		''' 
		Set boundary conditions for the given quantity q.
		Parameters:
			q (np.array): Quantity to set boundary conditions for
			boundary_type (str): "vertical" forces the quantity to be antisymmetric at the top and bottom boundaries, 
			                    "horizontal" forces the quantity to be antisymmetric at the left and right boundaries
		'''    
		### TODO: Step 2: Complete this function
		if boundary_type is None:
			# Copy adjacent cells to boundary
			q[0, :] = q[1, :]
			q[-1, :] = q[-2, :]
			q[:, 0] = q[:, 1]
			q[:, -1] = q[:, -2]
		elif boundary_type == 'vertical':
			# Set vertical boundaries to zero
			q[:, 0] = 0
			q[:, -1] = 0
		elif boundary_type == 'horizontal':
			# Set horizontal boundaries to zero
			q[0, :] = 0
			q[-1, :] = 0
		else:
			raise ValueError(f"Invalid boundary type: {boundary_type}")
		return q

	def advect_particles(self, dt):
		''' Advects particles using the current grid velocities. '''
		# TODO: STEP 1: Complete this function
		i, j, a, b = self.xy_to_ij(self.particles[:, 0], self.particles[:, 1])

		# print("this is curr_v", self.curr_v)

		v00 = self.curr_v[i, j, 1]  # y-component of velocity at bottom-left corner
		v10 = self.curr_v[i+1, j, 1]  # y-component of velocity at bottom-right corner
		v01 = self.curr_v[i, j+1, 1]  # y-component of velocity at top-left corner
		v11 = self.curr_v[i+1, j+1, 1]  # y-component of velocity at top-right corner

		b_v = (1-a)*(1-b)*v00 + (1-a)*b*v10 + a*(1-b)*v01 + a*b*v11  # bilinear interpolation

		v00 = self.curr_v[i, j, 0]
		v10 = self.curr_v[i+1, j, 0]
		v01 = self.curr_v[i, j+1, 0]
		v11 = self.curr_v[i+1, j+1, 0]

		a_v = (1-a)*(1-b)*v00 + (1-a)*b*v10 + a*(1-b)*v01 + a*b*v11

		# print("this is v00", v00)
		# print("this is v10", v10)
		# print("this is v01", v01)
		# print("this is v11", v11)



		# print("this is self.particles[:, 0]", self.particles[:, 0])
		# print("this is self.particles[:, 1]", self.particles[:, 1])

		self.particles[:, 0] += a_v * dt
		self.particles[:, 1] += b_v * dt

		# print("this is a_v", a_v)
		# print("this is b_v", b_v)

		self.particles[:, 0] = np.clip(self.particles[:, 0], self.xl, self.xh)
		self.particles[:, 1] = np.clip(self.particles[:, 1], self.yl, self.yh)

	def advect(self, q, dt, gv, boundary_type):
		'''
		Advects given quantities q given the current grid velocities gv.
			Parameters:
				q (np.array): Quantities to be advected
				dt (float): Time step
				gv (np.array): Grid velocities
				boundary_type (str): type of boundary condition to apply (None, "vertical", "horizontal")
			Returns (np.array): Advected quantities
		'''
		# TODO: STEP 4: Complete this function

		# dt0 = dt * self.iterations
		# i, j, a, b = self.xy_to_ij(self.particles[:, 0], self.particles[:, 1])

		# for k in range(self.iterations):
		# 	for l in range(self.iterations):
		# 		a[k,l] = k - dt0 * gv[k, l, 1]
		# 		b[k,l] = l - dt0 * gv[k, l, 0]
		# 		i = 0.5 if i < 0.5 else i

		#create meshgrid for the actual domain grid


		# d = q.copy()
		# d0 = q.copy()
		# # print("this is gv", gv)
		# u, v = gv[:,:,0], gv[:,:,1]
		# dt0 = dt * N

		# for i in range(1, self.ny+1):
		# 	for j in range(1, self.nx+1):
		# 		x = j - dt0 * v[i, j]
		# 		y = i - dt0 * u[i, j]
		# 		x = min(max(x, 0.5), self.nx + 0.5)
		# 		y = min(max(y, 0.5), self.ny + 0.5)
		# 		i0, j0 = int(y), int(x)
		# 		i1, j1 = i0 + 1, j0 + 1
		# 		s1, t1 = x - j0, y - i0
		# 		s0, t0 = 1 - s1, 1 - t1
		# 		d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])

		# self.set_boundary(d, boundary_type)



		d = q.copy()
		d0 = q.copy()
		u, v = gv[:,:,0], gv[:,:,1]

		I, J = np.ogrid[1:self.ny+1, 1:self.nx+1]

		X = J - dt*self.nx * v[1:self.ny+1, 1:self.nx+1]
		Y = I - dt*self.ny * u[1:self.ny+1, 1:self.nx+1]

		X = np.clip(X, 0.5, self.nx + 0.5)
		Y = np.clip(Y, 0.5, self.ny + 0.5)

		I0, J0 = np.floor(Y).astype(int), np.floor(X).astype(int)
		I1, J1 = I0 + 1, J0 + 1
		S1, T1 = X - J0, Y - I0
		S0, T0 = 1 - S1, 1 - T1

		d[1:self.ny+1, 1:self.nx+1] = S0 * (T0 * d0[I0, J0] + T1 * d0[I0, J1]) + S1 * (T0 * d0[I1, J0] + T1 * d0[I1, J1])

		self.set_boundary(d, boundary_type)
    	
		return d

	def step(self, dt):
		self.add_source_temperature(dt)
		#self.apply_temperature_force(dt)        
		self.velocity_step(dt)
		self.scalar_step(dt)
		self.advect_particles(dt)
		print("---------------------------------------------------------------------")

	def add_source_temperature(self, dt):
		print("add source temperature is being called")
		for source in self.sources:
			i,j,a,b = self.xy_to_ij(source.x, source.y)
			self.curr_tp[i,j] += (1-a)*(1-b)*source.strength
			self.curr_tp[i+1,j] += (1-a)*b*source.strength
			self.curr_tp[i,j+1] += a*(1-b)*source.strength
			self.curr_tp[i+1,j+1] += a*b*source.strength

	def apply_temperature_force(self, dt):
		'''Applies the buoyancy force to the velocity field.'''
		# TODO: STEP 9: Complete this function
		ref_temp = np.mean(self.curr_tp)

		# Compute the temperature delta
		temp_delta = self.curr_tp - ref_temp

		# Update the vertical velocity
		self.curr_v[:,:,1] += self.beta * dt * temp_delta
		
		# Set the boundary conditions
		self.set_boundary(self.curr_v[:,:,1], 'vertical')

	def scalar_step(self, dt):
		# TODO: STEP 5: Complete this function with a call to the diffuse and advect functions
		self.curr_tp = self.diffuse(self.curr_tp, self.kappa, dt, None)
		self.curr_tp = self.advect(self.curr_tp, dt, self.curr_v, None)

	def velocity_step(self, dt):
		# TODO: STEP 7: Complete this function with calls to the diffuse, project, and advect functions
		N = len(self.curr_tp) - 2
		visc = self.viscosity
		self.next_v[:,:,0] = self.diffuse(self.curr_v[:,:,0], visc, dt, "horizontal")
		self.next_v[:,:,1] = self.diffuse(self.curr_v[:,:,1], visc, dt, "vertical")

		self.curr_v[:,:,0] = self.next_v[:,:,0]
		self.curr_v[:,:,1] = self.next_v[:,:,1]

		self.project()

		self.next_v[:,:,0] = self.advect(self.curr_v[:,:,0], dt, self.curr_v, "horizontal")
		self.next_v[:,:,1] = self.advect(self.curr_v[:,:,1], dt, self.curr_v, "vertical")

		self.curr_v[:,:,0] = self.next_v[:,:,0]
		self.curr_v[:,:,1] = self.next_v[:,:,1]

		self.project()

	# Projection step to make the velocity field divergence free
	def project(self):
		'''Solves the pressure Poisson equation to make the velocity field divergence free.'''
		# TODO: STEP 6: Complete this function
		# Nx = 1/self.dx
		# Ny = 1/self.dy
		# hx = 1.0 / Nx
		# hy = 1.0 / Ny
		u, v = self.curr_v[:,:,0], self.curr_v[:,:,1]
		p = np.zeros((self.ny+2, self.nx+2))
		div = np.zeros((self.ny+2, self.nx+2))

		div[1:self.ny+1, 1:self.nx+1] = -0.5 * self.dx * (u[1:self.ny+1, 2:self.nx+2] - u[1:self.ny+1, :-2] + v[2:self.ny+2, 1:self.nx+1] - v[:-2, 1:self.nx+1])
		p[1:self.ny+1, 1:self.nx+1] = 0

		# for i in range(1, self.ny+1):
		# 	for j in range(1, self.nx+1):
		# 		div[i, j] = -0.5 * self.dx * (u[i, j+1] - u[i, j-1] + v[i+1, j] - v[i-1, j])
		# 		p[i, j] = 0

		self.set_boundary(div, None)
		self.set_boundary(p, None)

		for _ in range(20):
			for i in range(1, self.ny+1):
				for j in range(1, self.nx+1):
					p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1]) / 4

			self.set_boundary(p, None)

		# for i in range(1, self.ny+1):
		# 	for j in range(1, self.nx+1):
		# 		v[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j]) / self.dx
		# 		u[i, j] -= 0.5 * (p[i, j+1] - p[i, j-1]) / self.dx
			
		v[1:self.ny+1, 1:self.nx+1] -= 0.5 * (p[2:self.ny+2, 1:self.nx+1] - p[:-2, 1:self.nx+1]) / self.dx
		u[1:self.ny+1, 1:self.nx+1] -= 0.5 * (p[1:self.ny+1, 2:self.nx+2] - p[1:self.ny+1, :-2]) / self.dx

		self.set_boundary(u, "horizontal")
		self.set_boundary(v, "vertical")
		
		self.curr_v[:,:,0] = u
		self.curr_v[:,:,1] = v
		
	
	def render(self, time, frame_time):		
		self.ctx.clear(0,0,0)
		if self.reset_request:
			self.reset_request = False
			self.reset()
		if self.running or self.step_request:
			self.step_request = False
			self.step(self.dt)		
		if self.draw_grid_enabled: 
			self.draw_color_mesh()
		if self.draw_velocity_enabled: self.draw_velocity_lines()
		if self.draw_grid_lines_enabled: 
			self.grid_lines_vao.program['colour'] = (0.5, 0.5, 0.5, 0.25)
			self.grid_lines_vao.render()
		self.domain_box_vao.program['colour'] = (1, 1, 1, 1)
		self.domain_box_vao.render()
		self.draw_particles()
		for source in self.sources:
			self.prog2['colour'] = source.colour
			self.prog2['offset'] = (source.x, source.y)
			self.prog2['scale'] = (self.dx/4, self.dy/4)	
			self.star_vao.render()
		self.prog2['offset'] = (0,0) # reset to identity
		self.prog2['scale'] = (1,1)	# reset to identity	    

HelloWorld.run()