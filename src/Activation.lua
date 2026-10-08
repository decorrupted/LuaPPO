--!strict

--[[
	Activation.lua

	Activation functions for neural networks.

	Supported:
		- ReLU
		- Tanh

	Each activation supports:
		- forward(input)
		- backward(gradientOutput)

	The activation stores its forward-pass values so
	the derivative can be calculated during backpropagation.
]]

local Activation = {}
Activation.__index = Activation

export type ActivationType = "ReLU" | "Tanh"

export type ActivationLayer = {
	activationType: ActivationType,
	lastInput: {number}?,

	forward: (self: ActivationLayer, input: {number}) -> {number},
	backward: (self: ActivationLayer, gradientOutput: {number}) -> {number},
}

--------------------------------------------------
-- Constructor
--------------------------------------------------

function Activation.new(
	activationType: ActivationType
): ActivationLayer

	assert(
		type(activationType) == "string",
		"Activation.new: activationType must be a string"
	)

	assert(
		activationType == "ReLU"
			or activationType == "Tanh",
		"Activation.new: unsupported activation. Use 'ReLU' or 'Tanh'"
	)

	local self: ActivationLayer = setmetatable({
		activationType = activationType,
		lastInput = nil,
	}, Activation)

	return self
end

--------------------------------------------------
-- Input validation
--------------------------------------------------

function Activation:_validateInput(
	input: {number}
)

	assert(
		type(input) == "table",
		"Activation.forward: input must be a vector/table"
	)

	for i = 1, #input do
		assert(
			type(input[i]) == "number",
			string.format(
				"Activation.forward: input[%d] must be a number",
				i
			)
		)
	end
end

--------------------------------------------------
-- Forward
--------------------------------------------------

function Activation:forward(
	input: {number}
): {number}

	self:_validateInput(input)

	-- Store input because ReLU's derivative depends
	-- on whether the original value was positive.
	self.lastInput = table.create(#input)

	local output = table.create(#input)

	for i = 1, #input do
		local x = input[i]

		self.lastInput[i] = x

		if self.activationType == "ReLU" then

			if x > 0 then
				output[i] = x
			else
				output[i] = 0
			end

		elseif self.activationType == "Tanh" then

			output[i] = math.tanh(x)

		end
	end

	return output
end

--------------------------------------------------
-- Backward
--------------------------------------------------

function Activation:backward(
	gradientOutput: {number}
): {number}

	assert(
		self.lastInput ~= nil,
		"Activation.backward: forward must be called before backward"
	)

	assert(
		#gradientOutput == #self.lastInput,
		string.format(
			"Activation.backward: expected gradient of size %d, got %d",
			#self.lastInput,
			#gradientOutput
		)
	)

	local input = self.lastInput :: {number}

	local gradientInput = table.create(#input)

	for i = 1, #input do

		local x = input[i]
		local gradient = gradientOutput[i]

		if self.activationType == "ReLU" then

			-- ReLU:
			--
			-- f(x) = x       if x > 0
			--        0       otherwise
			--
			-- f'(x) = 1      if x > 0
			--         0      otherwise

			if x > 0 then
				gradientInput[i] = gradient
			else
				gradientInput[i] = 0
			end

		elseif self.activationType == "Tanh" then

			-- tanh derivative:
			--
			-- 1 - tanh(x)^2

			local tanhX = math.tanh(x)

			gradientInput[i] =
				gradient * (1 - tanhX * tanhX)

		end
	end

	return gradientInput
end

--------------------------------------------------
-- Reset stored state
--------------------------------------------------

function Activation:clear()
	self.lastInput = nil
end

return Activation
