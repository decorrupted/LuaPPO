--!strict

-- GAE.lua
-- Generalized Advantage Estimation for PPO.

local GAE = {}

--------------------------------------------------
-- Types
--------------------------------------------------

export type Experience = {
	observation: {number},
	action: {number},
	reward: number,
	value: number,
	logProb: number,

	terminated: boolean,
	truncated: boolean,

	nextValue: number,
}

export type GAEOutput = {
	advantages: {number},
	returns: {number},
}

--------------------------------------------------
-- Constants
--------------------------------------------------

local DEFAULT_GAMMA = 0.99
local DEFAULT_LAMBDA = 0.95

--------------------------------------------------
-- Validation helpers
--------------------------------------------------

local function isFinite(value: number): boolean
	return value == value
		and value < math.huge
		and value > -math.huge
end

local function validateHyperparameter(
	value: number,
	name: string
)
	assert(
		type(value) == "number"
			and isFinite(value)
			and value >= 0
			and value <= 1,
		name .. " must be a finite number between 0 and 1"
	)
end

--------------------------------------------------
-- Compute advantages and returns
--------------------------------------------------

function GAE.compute(
	experiences: {Experience},
	gamma: number?,
	lambdaValue: number?
): GAEOutput

	local discount = gamma
	if discount == nil then
		discount = DEFAULT_GAMMA
	end

	local gaeLambda = lambdaValue
	if gaeLambda == nil then
		gaeLambda = DEFAULT_LAMBDA
	end

	validateHyperparameter(discount, "gamma")
	validateHyperparameter(gaeLambda, "lambda")

	assert(
		type(experiences) == "table",
		"experiences must be a table"
	)

	local count = #experiences

	local advantages = table.create(count, 0)
	local returns = table.create(count, 0)

	local nextAdvantage = 0

	-- Work backward because each advantage depends
	-- on the advantage of the following transition.
	for t = count, 1, -1 do
		local experience = experiences[t]

		assert(
			type(experience) == "table",
			string.format("Experience %d must be a table", t)
		)

		local reward = experience.reward
		local value = experience.value
		local nextValue = experience.nextValue

		assert(
			type(reward) == "number" and isFinite(reward),
			string.format("Experience %d has invalid reward", t)
		)

		assert(
			type(value) == "number" and isFinite(value),
			string.format("Experience %d has invalid value", t)
		)

		assert(
			type(nextValue) == "number"
				and isFinite(nextValue),
			string.format("Experience %d has invalid nextValue", t)
		)

		assert(
			type(experience.terminated) == "boolean",
			string.format("Experience %d has invalid terminated flag", t)
		)

		assert(
			type(experience.truncated) == "boolean",
			string.format("Experience %d has invalid truncated flag", t)
		)

		--------------------------------------------------
		-- Bootstrap mask
		--------------------------------------------------

		-- A true terminal state has no future value.
		-- A time-limit truncation can still bootstrap
		-- from its final state's value estimate.
		local bootstrapMask = 1

		if experience.terminated then
			bootstrapMask = 0
		end

		--------------------------------------------------
		-- TD residual
		--------------------------------------------------

		local delta =
			reward
			+ discount * nextValue * bootstrapMask
			- value

		--------------------------------------------------
		-- Advantage recursion mask
		--------------------------------------------------

		-- Do not carry the next episode's advantage
		-- backward across a terminal or truncated
		-- episode boundary.
		local continuationMask = 1

		if experience.terminated or experience.truncated then
			continuationMask = 0
		end

		local advantage =
			delta
			+ discount
				* gaeLambda
				* continuationMask
				* nextAdvantage

		assert(
			isFinite(advantage),
			string.format("Computed invalid advantage at index %d", t)
		)

		local returnValue = advantage + value

		assert(
			isFinite(returnValue),
			string.format("Computed invalid return at index %d", t)
		)

		advantages[t] = advantage
		returns[t] = returnValue

		nextAdvantage = advantage
	end

	return {
		advantages = advantages,
		returns = returns,
	}
end

--------------------------------------------------
-- Convenience function for a RolloutBuffer
--------------------------------------------------

function GAE.computeFromBuffer(
	buffer: any,
	gamma: number?,
	lambdaValue: number?
): GAEOutput

	assert(
		type(buffer) == "table"
			and type(buffer.getAll) == "function",
		"computeFromBuffer expects a RolloutBuffer"
	)

	return GAE.compute(
		buffer:getAll(),
		gamma,
		lambdaValue
	)
end

return GAE
