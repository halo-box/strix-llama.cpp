import type { SidebarAction } from '$lib/enums';
import type { Component } from 'svelte';

/**
 * A single clickable action in the desktop sidebar icon strip.
 */
export interface DesktopIconStripItem {
	icon: Component;
	tooltip: string;
	route?: string;
	/** Page outside the UI (served by llama-server), opened in a new tab */
	externalHref?: string;
	/** Custom action handled by the sidebar, e.g. opening a new-chat tab */
	action?: SidebarAction;
	activeRouteId?: string;
	activeRoutePrefix?: string;
	activeUrlIncludes?: string;
	keys?: string[];
}
