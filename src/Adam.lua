--!strict

--[[
	Adam.lua

	Adam optimizer for the trainable Linear layers
	inside our Network.lua.

	Features:
		- First and second moment estimates
		- Bias correction
		- Configurable learning rate
		- Configurable beta1, beta2, and epsilon
		- Separate optimizer state for every layer
		- Weight and bias updates
		- Learning rate adjustment
		- Optimizer state reset

	Usage:

		local Adam = require(script.Parent.Adam)

		local optimizer = Adam.new(network, 0.001)

		network:zeroGrad()
		local output = network:forward(input)
		local gradientInput = network:backward(gradientOutput)

		optimizer:step()

	Do not call network:step() as well:
		Adam performs the parameter updates itself.
]]

local Adam = {}
Adam.__index = Adam

--------------------------------------------------
-- Types
--------------------------------------------------

export type AdamOptimizer = {
	network: any,

	learningRate: number,
	beta1: number,
	beta2: number,
	epsilon: number,
	timeStep: number,

	stateByLayer: {[any]: any},

	step: (self: AdamOptimizer) -> (),
	setLearningRate: (
		self: AdamOptimizer,
		learningRate: number
	) -> (),
	reset: (self: AdamOptimizer) -> (),
}

--------------------------------------------------
-- Helpers
--------------------------------------------------

local function validatePositiveNumber(
	value: number,
	name: string,
	allowZero: boolean?
)
	assert(
		type(value) == "number"
			and value == value
			and value < math.huge
			and value > -math.huge,
		name .. " must be a finite number"
	)

	if allowZero then
		assert(value >= 0, name .. " must be non-negative")
	else
		assert(value > 0, name .. " must be positive")
	end
end

local function validateBeta(
	value: number,
	name: string
)
	assert(
		type(value) == "number"
			and value == value
			and value >= 0
			and value < 1,
		name .. " must be in the range [0, 1)"
	)
end

local function createLayerState(layer: any): any
	local weights = layer.weights
	local bias = layer.bias

	local mWeights = table.create(#weights.data)
	local vWeights = table.create(#weights.data)

	for i = 1, #weights.data do
		mWeights[i] = table.create(weights.cols, 0)
		vWeights[i] = table.create(weights.cols, 0)
	end

	return {
		mWeights = mWeights,
		vWeights = vWeights,

		mBias = table.create(#bias, 0),
		vBias = table.create(#bias, 0),
	}
end

--------------------------------------------------
-- Constructor
--------------------------------------------------

function Adam.new(
	network: any,
	learningRate: number?,
	beta1: number?,
	beta2: number?,
	epsilon: number?
): AdamOptimizer

	assert(
		type(network) == "table"
			and type(network.layers) == "table",
		"Adam.new: network must contain a layers table"
	)

	local lr = learningRate or 0.001
	local b1 = beta1 or 0.9
	local b2 = beta2 or 0.999
	local eps = epsilon or 1e-8

	validatePositiveNumber(lr, "Adam.new: learningRate")
	validateBeta(b1, "Adam.new: beta1")
	validateBeta(b2, "Adam.new: beta2")
	validatePositiveNumber(eps, "Adam.new: epsilon")

	local self: AdamOptimizer = setmetatable({
		network = network,

		learningRate = lr,
		beta1 = b1,
		beta2 = b2,
		epsilon = eps,

		timeStep = 0,

		stateByLayer = {},
	}, Adam)

	-- Each trainable layer gets its own moment estimates.
	for _, layer in ipairs(network.layers) do
		if layer.weights ~= nil then
			assert(
				layer.gradWeights ~= nil
					and layer.bias ~= nil
					and layer.gradBias ~= nil,
				"Adam.new: trainable layer is missing parameters or gradients"
			)

			self.stateByLayer[layer] = createLayerState(layer)
		end
	end

	return self
end

--------------------------------------------------
-- Step
--------------------------------------------------

-- Apply Adam updates to all trainable layers.
--
-- Call network:backward(gradient) first.
-- Call network:zeroGrad() before accumulating
-- gradients for the next optimization step.

function Adam:step()
	self.timeStep += 1

	local t = self.timeStep
	local beta1 = self.beta1
	local beta2 = self.beta2
	local learningRate = self.learningRate
	local epsilon = self.epsilon

	-- Bias correction factors.
	local correction1 = 1 - beta1 ^ t
	local correction2 = 1 - beta2 ^ t

	for layer, state in pairs(self.stateByLayer) do
		local weights = layer.weights.data
		local gradients = layer.gradWeights.data

		--------------------------------------------------
		-- Update weights
		--------------------------------------------------

		for i = 1, layer.weights.rows do
			for j = 1, layer.weights.cols do
				local gradient = gradients[i][j]

				local m = beta1 * state.mWeights[i][j]
					+ (1 - beta1) * gradient

				local v = beta2 * state.vWeights[i][j]
					+ (1 - beta2) * gradient * gradient

				state.mWeights[i][j] = m
				state.vWeights[i][j] = v

				local mHat = m / correction1
				local vHat = v / correction2

				weights[i][j] -= learningRate
					* mHat
					/ (math.sqrt(vHat) + epsilon)
			end
		end

		--------------------------------------------------
		-- Update biases
		--------------------------------------------------

		for i = 1, #layer.bias do
			local gradient = layer.gradBias[i]

			local m = beta1 * state.mBias[i]
				+ (1 - beta1) * gradient

			local v = beta2 * state.vBias[i]
				+ (1 - beta2) * gradient * gradient

			state.mBias[i] = m
			state.vBias[i] = v

			local mHat = m / correction1
			local vHat = v / correction2

			layer.bias[i] -= learningRate
				* mHat
				/ (math.sqrt(vHat) + epsilon)
		end
	end
end

--------------------------------------------------
-- Set learning rate
--------------------------------------------------

function Adam:setLearningRate(
	learningRate: number
)
	validatePositiveNumber(
		learningRate,
		"Adam.setLearningRate: learningRate"
	)

	self.learningRate = learningRate
end

--------------------------------------------------
-- Reset optimizer state
--------------------------------------------------

-- Clears moment estimates and resets the step counter.
-- Does not change the network's parameters.

function Adam:reset()
	self.timeStep = 0
	self.stateByLayer = {}

	for _, layer in ipairs(self.network.layers) do
		if layer.weights ~= nil then
			self.stateByLayer[layer] = createLayerState(layer)
		end
	end
end

return Adam
