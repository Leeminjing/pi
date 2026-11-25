// Tool parks: what `execute()` returns instead of a result to sleep or wait, its shape check, and how the Harness tells
// a base tool's parks from a wrapper's.
import type { Context } from "@earendil-works/chord/context";
import type {
	ToolCancelApi,
	ToolCancelPark,
	ToolExecutionApi,
	ToolExecutionResult,
	ToolPark,
	ToolRegistration,
} from "./types.ts";

/** Whether `execute()` returned a park rather than a result. */
export function isPark(returned: ToolExecutionResult | ToolPark): returned is ToolPark {
	// Tool code is a trust boundary: any `kind` makes a park, and `parkError` refuses one that is neither.
	return returned.kind !== undefined;
}

/**
 * The parks a wrapped base returned in one run of a tool, carried by that run's api under this key. An own enumerable
 * symbol property, so a wrapper that spreads the api, as wrappers do, passes it on to the base.
 */
export const BASE_PARKS: unique symbol = Symbol("pi.tool.baseParks");

/** A run's api with the parks its wrapped base returned. */
type ParkCarrier = { readonly [BASE_PARKS]?: Set<ToolPark> };

/** The parks the wrapped base returned during this cancellation attempt. */
export const BASE_CANCEL_PARKS: unique symbol = Symbol("pi.tool.baseCancelParks");

export type CancelParkCarrier = { readonly [BASE_CANCEL_PARKS]?: Set<ToolCancelPark> };

/** Fields of a result; a park carrying one of them is invalid. */
const RESULT_FIELDS = ["output", "structuredOutput", "isError", "details", "diagnostics", "usage", "control"] as const;

/**
 * Why a park is invalid: wrong kind, a result field, or a wrapper returning a park absent from this attempt's
 * `baseParks`. Rules on what it waits for are the commit's.
 */
export function parkError(
	tool: ToolRegistration,
	park: object,
	baseParks: ReadonlySet<object> | undefined,
): string | undefined {
	// Tool code is a trust boundary for the mixed objects its types forbid.
	const kind = (park as { readonly kind?: unknown }).kind;
	if (kind !== "sleep" && kind !== "wait") return `a park's kind is sleep or wait, not ${String(kind)}`;
	const field = RESULT_FIELDS.find((name) => name in park);
	if (field !== undefined) return `a park has no result field, but this one has ${field}`;
	if (composites.has(tool) && baseParks?.has(park) !== true) {
		return "a wrapper returned a park its base did not return";
	}
	return undefined;
}

/** Validate a cancellation return at the tool boundary, including its provenance for this attempt. */
export function cancelParkError(tool: ToolRegistration, park: unknown, api: CancelParkCarrier): string | undefined {
	if (typeof park !== "object" || park === null || !("kind" in park) || park.kind !== "sleep") {
		return "a cancellation park must be a sleep";
	}
	return parkError(tool, park, api[BASE_CANCEL_PARKS]);
}

/** Composite tools a wrapper built; their parks must come from their base. */
const composites = new WeakSet<ToolRegistration>();

/**
 * Apply `wrap` to `tool` as tool wrappers apply (spec suspendable §9): the first wrapper of a base gets a copy whose
 * `execute()` and `cancel()` record their parks in separate attempt-local sets carried by the api.
 */
export function wrapComposite<Tool extends ToolRegistration>(tool: Tool, wrap: (tool: Tool) => Tool): Tool {
	const cancel = tool.cancel;
	const base: Tool = composites.has(tool)
		? tool
		: {
				...tool,
				execute: async (args, api: ToolExecutionApi & ParkCarrier, context) => {
					const returned = await tool.execute(args, api, context);
					if (isPark(returned)) api[BASE_PARKS]?.add(returned);
					return returned;
				},
				...(cancel === undefined
					? {}
					: {
							cancel: async (args: unknown, api: ToolCancelApi & CancelParkCarrier, context: Context) => {
								const returned = await cancel.call(tool, args, api, context);
								if (returned !== undefined) api[BASE_CANCEL_PARKS]?.add(returned);
								return returned;
							},
						}),
			};
	const composite = wrap(base);
	composites.add(composite);
	return composite;
}
