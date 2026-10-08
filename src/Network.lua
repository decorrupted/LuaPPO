--!strict

--[[
	Network.lua

	A lightweight feed-forward neural network.

	Example:

		local network = Network.new({
			{ inputSize = 8, outputSize = 16 },
			{ activation = "Tanh" },
			{ inputSize = 16, outputSize = 16 },
			{ activation = "Tanh" },
			{ inputSize = 16, outputSize = 4 },
		})

		local output = network:forward(input)

		local gradientInput = network:backward(gradientOutput)

		network:zeroGrad()
		network:step(learningRate)

	This network is intentionally simple.

	It supports:
		- Linear layers
		- ReLU activations
		- Tanh activations
		- Forward propagation
		- Backpropagation
		- Gradient clearing
		- SGD parameter updates

	Later, this will become the foundation for
	the PPO actor and critic networks.
]]

local Linear = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/Linear.lua"))()
local Activation = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/Activation.lua"))()

local Network = {}
Network.__index = Network

--------------------------------------------------
-- Types
--------------------------------------------------

export type LayerConfig =
	{
		inputSize: number,
		outputSize: number,
	}
	|
	{
		activation: Activation.ActivationType,
	}

export type Layer =
	Linear.LinearLayer
	|
	Activation.ActivationLayer

export type NetworkType = {
	layers: {Layer},

	lastOutput: {number}?,

	forward: (
		self: NetworkType,
		input: {number}
	) -> {number},

	backward: (
		self: NetworkType,
		gradientOutput: {number}
	) -> {number},

	zeroGrad: (self: NetworkType) -> (),
	step: (self: NetworkType, learningRate: number) -> (),
	clear: (self: NetworkType) -> (),
	getParameterCount: (self: NetworkType) -> number,
}

--------------------------------------------------
-- Constructor
--------------------------------------------------

function Network.new(
	configs: {LayerConfig}
): NetworkType

	assert(
		type(configs) == "table",
		"Network.new: configs must be a table"
	)

	assert(
		#configs > 0,
		"Network.new: network must contain at least one layer"
	)

	local layers: {Layer} = {}

	local expectedInputSize: number? = nil

	for index, config in ipairs(configs) do

		--------------------------------------------------
		-- Linear layer
		--------------------------------------------------

		if config.inputSize ~= nil
			or config.outputSize ~= nil then

			assert(
				config.inputSize ~= nil
					and config.outputSize ~= nil,
				string.format(
					"Network.new: Linear layer %d requires inputSize and outputSize",
					index
				)
			)

			local inputSize = config.inputSize :: number
			local outputSize = config.outputSize :: number

			-- If there is a previous layer, its output
			-- must match this layer's input.
			if expectedInputSize ~= nil then

				assert(
					inputSize == expectedInputSize,
					string.format(
						"Network.new: layer %d expects %d inputs, previous layer outputs %d",
						index,
						inputSize,
						expectedInputSize
					)
				)

			end

			local layer = Linear.new(
				inputSize,
				outputSize
			)

			table.insert(
				layers,
				layer
			)

			expectedInputSize = outputSize

		--------------------------------------------------
		-- Activation layer
		--------------------------------------------------

		elseif config.activation ~= nil then

			assert(
				expectedInputSize ~= nil,
				string.format(
					"Network.new: activation layer %d cannot be the first layer",
					index
				)
			)

			local activationType =
				config.activation :: Activation.ActivationType

			local layer = Activation.new(
				activationType
			)

			table.insert(
				layers,
				layer
			)

		--------------------------------------------------
		-- Invalid layer
		--------------------------------------------------

		else

			error(
				string.format(
					"Network.new: invalid layer configuration at index %d",
					index
				)
			)

		end
	end

	local self: NetworkType = setmetatable({
		layers = layers,
		lastOutput = nil,
	}, Network)

	return self
end

--------------------------------------------------
-- Forward
--------------------------------------------------

function Network:forward(
	input: {number}
): {number}

	assert(
		type(input) == "table",
		"Network.forward: input must be a vector/table"
	)

	local current = input

	for _, layer in ipairs(self.layers) do

		current = layer:forward(current)

	end

	self.lastOutput = current

	return current
end

--------------------------------------------------
-- Backward
--------------------------------------------------

-- gradientOutput is dL/dY.
--
-- The gradient is passed backwards through
-- every layer in reverse order.
--
-- Returns:
--	gradientInput = dL/dX

function Network:backward(
	gradientOutput: {number}
): {number}

	assert(
		type(gradientOutput) == "table",
		"Network.backward: gradientOutput must be a vector/table"
	)

	assert(
		self.lastOutput ~= nil,
		"Network.backward: forward must be called before backward"
	)

	local currentGradient = gradientOutput

	-- Backpropagation goes from output → input.
	for i = #self.layers, 1, -1 do

		currentGradient =
			self.layers[i]:backward(currentGradient)

	end

	return currentGradient
end

--------------------------------------------------
-- Zero gradients
--------------------------------------------------

function Network:zeroGrad()

	for _, layer in ipairs(self.layers) do

		-- Only Linear layers contain trainable
		-- parameters and gradients.
		if layer.weights ~= nil then

			layer:zeroGrad()

		end

	end
end

--------------------------------------------------
-- Parameter update
--------------------------------------------------

function Network:step(
	learningRate: number
)

	assert(
		type(learningRate) == "number",
		"Network.step: learningRate must be a number"
	)

	assert(
		learningRate >= 0,
		"Network.step: learningRate must be non-negative"
	)

	for _, layer in ipairs(self.layers) do

		if layer.weights ~= nil then

			layer:step(learningRate)

		end

	end
end

--------------------------------------------------
-- Clear cached forward-pass state
--------------------------------------------------

function Network:clear()

	self.lastOutput = nil

	for _, layer in ipairs(self.layers) do

		if layer.clear ~= nil then

			layer:clear()

		else

			-- Linear layers use lastInput.
			layer.lastInput = nil

		end

	end
end

--------------------------------------------------
-- Parameter count
--------------------------------------------------

function Network:getParameterCount(): number

	local count = 0

	for _, layer in ipairs(self.layers) do

		if layer.getParameterCount ~= nil then

			count += layer:getParameterCount()

		end

	end

	return count
end

--------------------------------------------------
-- Return
--------------------------------------------------

return Network