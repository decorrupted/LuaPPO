--[[
	Linear.lua

	A fully connected neural-network layer:

		y = W*x + b

	Where:
		W = outputSize x inputSize
		x = inputSize
		b = outputSize
		y = outputSize

	This module supports:
		- Dimension validation
		- He initialization
		- Forward pass
		- Backward pass
		- Weight/bias gradients
		- Parameter access

	Designed for the PPO neural network we will build later.
]]

local Matrix = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/Matrix.lua"))()

local Linear = {}
Linear.__index = Linear

export type LinearLayer = {
	inputSize: number,
	outputSize: number,

	weights: Matrix.Mat,
	bias: Matrix.Vector,

	-- Saved during forward pass.
	lastInput: Matrix.Vector?,

	-- Gradients.
	gradWeights: Matrix.Mat,
	gradBias: Matrix.Vector,
}

--------------------------------------------------
-- Constructor
--------------------------------------------------

function Linear.new(
	inputSize: number,
	outputSize: number
): LinearLayer

	assert(
		type(inputSize) == "number",
		"Linear.new: inputSize must be a number"
	)

	assert(
		type(outputSize) == "number",
		"Linear.new: outputSize must be a number"
	)

	assert(
		inputSize >= 1 and inputSize % 1 == 0,
		"Linear.new: inputSize must be a positive integer"
	)

	assert(
		outputSize >= 1 and outputSize % 1 == 0,
		"Linear.new: outputSize must be a positive integer"
	)

	-- He initialization is a good default for
	-- ReLU-based neural networks.
	local weights = Matrix.he(outputSize, inputSize)

	local bias = Matrix.vector(outputSize, 0)

	local gradWeights = Matrix.zeros(outputSize, inputSize)
	local gradBias = Matrix.vector(outputSize, 0)

	local self: LinearLayer = setmetatable({
		inputSize = inputSize,
		outputSize = outputSize,

		weights = weights,
		bias = bias,

		lastInput = nil,

		gradWeights = gradWeights,
		gradBias = gradBias,
	}, Linear) 

	return self
end

--------------------------------------------------
-- Input validation
--------------------------------------------------

function Linear:_validateInput(
	input: Matrix.Vector
)

	assert(
		type(input) == "table",
		"Linear.forward: input must be a vector/table"
	)

	assert(
		#input == self.inputSize,
		string.format(
			"Linear.forward: expected %d inputs, got %d",
			self.inputSize,
			#input
		)
	)

	for i = 1, self.inputSize do
		assert(
			type(input[i]) == "number",
			string.format(
				"Linear.forward: input[%d] must be a number",
				i
			)
		)
	end
end

--------------------------------------------------
-- Forward pass
--------------------------------------------------

function Linear:forward(
	input: Matrix.Vector
): Matrix.Vector

	self:_validateInput(input)

	-- Save input for backpropagation.
	self.lastInput = Matrix.copyVector(input)

	-- W*x
	local output = Matrix.matVec(
		self.weights,
		input
	)

	-- + b
	for i = 1, self.outputSize do
		output[i] += self.bias[i]
	end

	return output
end

--------------------------------------------------
-- Backward pass
--------------------------------------------------

-- gradientOutput is dL/dY
--
-- Returns:
--	gradientInput = dL/dX
--
-- Also calculates:
--	gradientWeights = dL/dW
--	gradientBias    = dL/db

function Linear:backward(
	gradientOutput: Matrix.Vector
): Matrix.Vector

	assert(
		self.lastInput ~= nil,
		"Linear.backward: forward must be called before backward"
	)

	assert(
		#gradientOutput == self.outputSize,
		string.format(
			"Linear.backward: expected gradient of size %d, got %d",
			self.outputSize,
			#gradientOutput
		)
	)

	local input = self.lastInput :: Matrix.Vector

	--------------------------------------------------
	-- dL/dW
	--
	-- gradientOutput[i] * input[j]
	--------------------------------------------------

	for i = 1, self.outputSize do
		local gradient = gradientOutput[i]
		local gradRow = self.gradWeights.data[i]

		for j = 1, self.inputSize do
			gradRow[j] = gradient * input[j]
		end
	end

	--------------------------------------------------
	-- dL/db
	--------------------------------------------------

	for i = 1, self.outputSize do
		self.gradBias[i] = gradientOutput[i]
	end

	--------------------------------------------------
	-- dL/dX
	--
	-- W^T * gradientOutput
	--------------------------------------------------

	local gradientInput = table.create(self.inputSize, 0)

	for j = 1, self.inputSize do
		local sum = 0

		for i = 1, self.outputSize do
			sum += self.weights.data[i][j] * gradientOutput[i]
		end

		gradientInput[j] = sum
	end

	return gradientInput
end

--------------------------------------------------
-- Zero gradients
--------------------------------------------------

function Linear:zeroGrad()
	for i = 1, self.outputSize do

		local weightRow = self.gradWeights.data[i]

		for j = 1, self.inputSize do
			weightRow[j] = 0
		end

		self.gradBias[i] = 0
	end
end

--------------------------------------------------
-- Parameter update
--------------------------------------------------

-- This is intentionally a simple SGD update.
--
-- We will implement Adam separately later.
--
-- parameter -= learningRate * gradient

function Linear:step(learningRate: number)

	assert(
		type(learningRate) == "number",
		"Linear.step: learningRate must be a number"
	)

	for i = 1, self.outputSize do

		local weights = self.weights.data[i]
		local gradients = self.gradWeights.data[i]

		for j = 1, self.inputSize do
			weights[j] -= learningRate * gradients[j]
		end

		self.bias[i] -= learningRate * self.gradBias[i]
	end
end

--------------------------------------------------
-- Parameter information
--------------------------------------------------

function Linear:getParameterCount(): number
	return (self.inputSize * self.outputSize)
		+ self.outputSize
end

function Linear:getParameters()
	return {
		weights = self.weights,
		bias = self.bias,
	}
end

function Linear:getGradients()
	return {
		weights = self.gradWeights,
		bias = self.gradBias,
	}
end

return Linear