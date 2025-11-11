"""Coordinator for Yasno outages integration."""

from __future__ import annotations

import datetime
import logging
from contextlib import suppress
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.components.calendar import CalendarEvent
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.translation import async_get_translations
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_utils

from .api import OutageEvent, OutageEventType, YasnoOutagesApi
from .const import (
    API_STATUS_EMERGENCY_SHUTDOWNS,
    API_STATUS_SCHEDULE_APPLIES,
    API_STATUS_WAITING_FOR_SCHEDULE,
    CONF_CALENDAR,
    CONF_GROUP,
    CONF_PROVIDER,
    CONF_REGION,
    DOMAIN,
    EVENT_NAME_OUTAGE,
    PROVIDER_DTEK_FULL,
    PROVIDER_DTEK_SHORT,
    STATE_NORMAL,
    STATE_OUTAGE,
    STATE_STATUS_EMERGENCY_SHUTDOWNS,
    STATE_STATUS_SCHEDULE_APPLIES,
    STATE_STATUS_WAITING_FOR_SCHEDULE,
    TRANSLATION_KEY_EVENT_OUTAGE,
    UPDATE_INTERVAL,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .data import YasnoOutagesConfigEntry

LOGGER = logging.getLogger(__name__)

TIMEFRAME_TO_CHECK = timedelta(hours=24)


class YasnoOutagesCoordinator(DataUpdateCoordinator):
    """Coordinator for Yasno Outages data."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: YasnoOutagesConfigEntry,
        api: YasnoOutagesApi,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=UPDATE_INTERVAL),
        )
        self.config_entry = config_entry
        self.api = api
        self.group = config_entry.options.get(
            CONF_GROUP, config_entry.data.get(CONF_GROUP)
        )
        self.region = config_entry.options.get(
            CONF_REGION, config_entry.data.get(CONF_REGION)
        )
        self.provider = config_entry.options.get(
            CONF_PROVIDER, config_entry.data.get(CONF_PROVIDER)
        )
        self.region_id = None
        self.provider_id = None
        self._provider_name = None  # Cached provider name

    @property
    def event_name_map(self) -> dict:
        """Return a mapping of event names to translations."""
        return {
            EVENT_NAME_OUTAGE: self.translations.get(TRANSLATION_KEY_EVENT_OUTAGE),
        }

    @property
    def status_state_map(self) -> dict:
        """Return a mapping of status names to translations."""
        return {
            API_STATUS_SCHEDULE_APPLIES: STATE_STATUS_SCHEDULE_APPLIES,
            API_STATUS_WAITING_FOR_SCHEDULE: STATE_STATUS_WAITING_FOR_SCHEDULE,
            API_STATUS_EMERGENCY_SHUTDOWNS: STATE_STATUS_EMERGENCY_SHUTDOWNS,
        }

    async def _resolve_ids(self) -> None:
        """Resolve region and provider IDs from names."""
        if not self.api.regions_data:
            await self.api.fetch_regions()

        if self.region:
            region_data = self.api.get_region_by_name(self.region)
            if region_data:
                self.region_id = region_data["id"]
                if self.provider:
                    provider_data = self.api.get_provider_by_name(
                        self.region, self.provider
                    )
                    if provider_data:
                        self.provider_id = provider_data["id"]
                        # Cache the provider name for device naming
                        self._provider_name = provider_data["name"]

    async def _async_update_data(self) -> None:
        """Fetch data from new Yasno API."""
        LOGGER.debug("Starting _async_update_data")
        try:
            await self.async_fetch_translations()

            # Resolve IDs if not already resolved
            if self.region_id is None or self.provider_id is None:
                await self._resolve_ids()

                # Update API with resolved IDs
                self.api = YasnoOutagesApi(
                    region_id=self.region_id,
                    provider_id=self.provider_id,
                    group=self.group,
                )

            # Fetch outages data (now async with aiohttp, not blocking)
            await self.api.fetch_data()

            # Sync events to external calendar if configured
            await self.async_sync_events_to_calendar()
            LOGGER.debug("Calendar sync completed")

            # Schedule a delayed sync in case calendar wasn't ready yet
            calendar_entity = self.config_entry.options.get(
                CONF_CALENDAR, self.config_entry.data.get(CONF_CALENDAR)
            )
            if calendar_entity:
                self.hass.loop.call_later(10, self._delayed_sync)
                self.hass.loop.call_later(30, self._delayed_sync)
                self.hass.loop.call_later(60, self._delayed_sync)
        except Exception:
            LOGGER.exception("Error in _async_update_data")

    async def async_fetch_translations(self) -> None:
        """Fetch translations."""
        self.translations = await async_get_translations(
            self.hass,
            self.hass.config.language,
            "common",
            [DOMAIN],
        )

    def _get_next_event_of_type(self, state_type: str) -> CalendarEvent | None:
        """Get the next event of a specific type."""
        now = dt_utils.now()
        # Sort events to handle multi-day spanning events correctly
        next_events = sorted(
            self.get_events_between(
                now,
                now + TIMEFRAME_TO_CHECK,
            ),
            key=lambda _: _.start,
        )
        LOGGER.debug("Next events: %s", next_events)
        for event in next_events:
            if self._event_to_state(event) == state_type and event.start > now:
                return event
        return None

    @property
    def next_planned_outage(self) -> datetime.date | datetime.datetime | None:
        """Get the next planned outage time."""
        event = self._get_next_event_of_type(STATE_OUTAGE)
        LOGGER.debug("Next planned outage: %s", event)
        return event.start if event else None

    @property
    def next_connectivity(self) -> datetime.date | datetime.datetime | None:
        """Get next connectivity time."""
        current_event = self.get_current_event()
        current_state = self._event_to_state(current_event)

        # If currently in outage state, return when it ends
        if current_state == STATE_OUTAGE:
            return current_event.end if current_event else None

        # Otherwise, return the end of the next outage
        event = self._get_next_event_of_type(STATE_OUTAGE)
        LOGGER.debug("Next connectivity: %s", event)
        return event.end if event else None

    @property
    def current_state(self) -> str:
        """Get the current state."""
        event = self.get_current_event()
        return self._event_to_state(event)

    @property
    def schedule_updated_on(self) -> datetime.datetime | None:
        """Get the schedule last updated timestamp."""
        return self.api.get_updated_on()

    @property
    def status_today(self) -> str | None:
        """Get the status for today."""
        return self.status_state_map.get(self.api.get_status_today())

    @property
    def status_tomorrow(self) -> str | None:
        """Get the status for tomorrow."""
        return self.status_state_map.get(self.api.get_status_tomorrow())

    @property
    def region_name(self) -> str:
        """Get the configured region name."""
        return self.region or ""

    @property
    def provider_name(self) -> str:
        """Get the configured provider name."""
        # Return cached name if available (but apply simplification first)
        if self._provider_name:
            return self._simplify_provider_name(self._provider_name)

        # Fallback to lookup if not cached yet
        if not self.api.regions_data:
            return ""

        region_data = self.api.get_region_by_name(self.region)
        if not region_data:
            return ""

        providers = region_data.get("dsos", [])
        for provider in providers:
            if (provider_name := provider.get("name", "")) == self.provider:
                # Cache the simplified name
                self._provider_name = provider_name
                return self._simplify_provider_name(provider_name)

        return ""

    def get_current_event(self) -> CalendarEvent | None:
        """Get the event at the present time."""
        return self.get_event_at(dt_utils.now())

    def get_event_at(self, at: datetime.datetime) -> CalendarEvent | None:
        """Get the event at a given time."""
        event = self.api.get_current_event(at)
        return self._get_calendar_event(event)

    def get_events_between(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
    ) -> list[CalendarEvent]:
        """Get all events."""
        events = self.api.get_events(start_date, end_date)
        return [self._get_calendar_event(event) for event in events]

    def _get_calendar_event(
        self,
        event: OutageEvent | None,
    ) -> CalendarEvent | None:
        """Transform an event into a CalendarEvent."""
        if not event:
            return None

        event_type = event.event_type.value
        summary = self.event_name_map.get(event_type)

        # Generate stable UID based on event time and group
        uid = f"{self.group}-{int(event.start.timestamp())}"

        output = CalendarEvent(
            summary=summary,
            start=event.start,
            end=event.end,
            description=event_type,
            uid=uid,
        )
        LOGGER.debug("Calendar Event: %s", output)
        return output

    def _event_to_state(self, event: CalendarEvent | None) -> str | None:
        if not event:
            return STATE_NORMAL

        # Event state is determined by whether it's a definite outage
        # Check description field which contains the event type
        if event.description == OutageEventType.DEFINITE.value:
            return STATE_OUTAGE

        LOGGER.warning("Unknown event type: %s", event.description)
        return STATE_NORMAL

    def _simplify_provider_name(self, provider_name: str) -> str:
        """Simplify provider names for cleaner display in device names."""
        # Replace long DTEK provider names with just "ДТЕК"
        if PROVIDER_DTEK_FULL in provider_name.upper():
            return PROVIDER_DTEK_SHORT

        # Add more provider simplifications here as needed
        return provider_name

    async def async_sync_events_to_calendar(self) -> None:
        """Sync events to calendar."""
        LOGGER.info("=== STARTING CALENDAR SYNC ===")

        # Continue with normal sync logic...
        calendar_entity = self.config_entry.options.get(
            CONF_CALENDAR,
            self.config_entry.data.get(CONF_CALENDAR),
        )
        LOGGER.debug("Calendar entity: %r", calendar_entity)
        LOGGER.debug("Calendar entity from config: %s", calendar_entity)

        if not calendar_entity:
            LOGGER.debug("No calendar entity configured for sync")
            return

        try:
            now = dt_utils.now()
            end_date = now + timedelta(days=8)
            events = self.get_events_between(now, end_date)

            LOGGER.info(
                "Syncing %d events to calendar %s", len(events), calendar_entity
            )

            # Get calendar object
            calendar_obj = None
            entity = self.hass.states.get(calendar_entity)
            LOGGER.debug("Calendar entity state: %s", entity)
            if entity:
                calendar_obj = self.hass.data["calendar"].get_entity(calendar_entity)
                LOGGER.debug("Calendar object from get_entity: %s", calendar_obj)

            if not calendar_obj:
                LOGGER.debug("Calendar object for %s not found", calendar_entity)
                return

            LOGGER.debug("Calendar object type: %s", type(calendar_obj))
            LOGGER.debug(
                "Calendar object has async_get_events: %s",
                hasattr(calendar_obj, "async_get_events"),
            )

            # Clean up old events first
            LOGGER.debug("About to call cleanup for %d events", len(events))
            await self._cleanup_old_events(calendar_obj, events)
            LOGGER.debug("Cleanup completed")

            # Sync new events
            for event in events:
                await self._sync_event_to_calendar(calendar_obj, event)

        except Exception:
            LOGGER.exception("Failed to sync events to calendar")

    async def _cleanup_old_events(
        self, calendar_obj: Any, current_events: list[CalendarEvent]
    ) -> None:
        """Clean up old Yasno events from calendar."""
        LOGGER.debug("Starting cleanup of old Yasno events")
        try:
            # Get all events from calendar for the next week
            now = dt_utils.now()
            end_date = now + timedelta(days=8)
            try:
                all_events = await calendar_obj.async_get_events(
                    self.hass, now, end_date
                )
                LOGGER.debug(
                    "Successfully retrieved %d events from calendar",
                    len(all_events),
                )
            except (AttributeError, TypeError, ValueError):
                LOGGER.exception("Failed to get events from calendar")
                return
            for cal_event in all_events[:5]:  # Log first 5 events for debugging
                LOGGER.debug(
                    "Calendar event: uid=%s, summary=%s, description=%s",
                    cal_event.uid,
                    cal_event.summary,
                    getattr(cal_event, "description", None),
                )

            # Create set of current Yasno event signatures
            # (summary + start time + end time)
            current_signatures = {
                (event.summary, event.start, event.end) for event in current_events
            }

            LOGGER.debug(
                "Found %d total events in calendar, %d current Yasno events",
                len(all_events),
                len(current_signatures),
            )

            deleted_count = 0
            yasno_events_found = 0
            for cal_event in all_events:
                # Check if this is a Yasno event (has our summary and description)
                yasno_summary = self.event_name_map.get(EVENT_NAME_OUTAGE)
                yasno_description = OutageEventType.DEFINITE.value
                event_desc = getattr(cal_event, "description", None)
                is_yasno_event = (
                    cal_event.summary == yasno_summary
                    and event_desc == yasno_description
                )

                if is_yasno_event:
                    yasno_events_found += 1
                    # For Yasno events, we delete ALL of them and recreate fresh
                    # This prevents accumulation of duplicates
                    with suppress(HomeAssistantError, ValueError):
                        await calendar_obj.async_delete_event(cal_event.uid)
                        LOGGER.debug("Deleted Yasno event: %s", cal_event.uid)
                        deleted_count += 1

            LOGGER.debug(
                "Yasno events analysis: found %d Yasno events, deleted %d old ones",
                yasno_events_found,
                deleted_count,
            )

        except Exception:
            LOGGER.exception("Failed to cleanup old events")

    async def _sync_event_to_calendar(
        self, calendar_obj: Any, event: CalendarEvent
    ) -> None:
        """Sync a single event to the external calendar."""
        try:
            # Try to create the event
            await calendar_obj.async_create_event(
                summary=event.summary,
                description=event.description,
                dtstart=event.start,
                dtend=event.end,
                uid=event.uid,
            )
            LOGGER.debug(
                "Synced event to calendar: %s (uid: %s)", event.summary, event.uid
            )
        except (AttributeError, TypeError, ValueError) as exc:
            LOGGER.debug(
                "Could not sync event to calendar: %s (uid: %s) - %s",
                event.summary,
                event.uid,
                exc,
                exc_info=True,
            )

    def _delayed_sync(self) -> None:
        """Delayed sync attempt for calendar events."""
        self.hass.create_task(self.async_sync_events_to_calendar())
