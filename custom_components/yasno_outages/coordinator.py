"""Coordinator for Yasno outages integration."""

import datetime
import logging

from homeassistant.components.calendar import CalendarEvent
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.translation import async_get_translations
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_utils

from .api import OutageEvent, OutageEventType, YasnoOutagesApi
from .const import (
    API_STATUS_EMERGENCY_SHUTDOWNS,
    API_STATUS_SCHEDULE_APPLIES,
    API_STATUS_WAITING_FOR_SCHEDULE,
    CALENDAR_SYNC_TIME_TOLERANCE,
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

LOGGER = logging.getLogger(__name__)

TIMEFRAME_TO_CHECK = datetime.timedelta(hours=24)


class YasnoOutagesCoordinator(DataUpdateCoordinator):
    """Class to manage fetching Yasno outages data."""

    config_entry: ConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        api: YasnoOutagesApi,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            LOGGER,
            name=DOMAIN,
            update_interval=datetime.timedelta(minutes=UPDATE_INTERVAL),
        )
        self.hass = hass
        self.config_entry = config_entry
        self.translations = {}

        # Get configuration values
        self.region = config_entry.options.get(
            CONF_REGION,
            config_entry.data.get(CONF_REGION),
        )
        self.provider = config_entry.options.get(
            CONF_PROVIDER,
            config_entry.data.get(CONF_PROVIDER),
        )
        self.group = config_entry.options.get(
            CONF_GROUP,
            config_entry.data.get(CONF_GROUP),
        )

        if not self.region:
            region_required_msg = (
                "Region not set in configuration - this should not happen "
                "with proper config flow"
            )
            region_error = "Region configuration is required"
            LOGGER.error(region_required_msg)
            raise ValueError(region_error)

        if not self.provider:
            provider_required_msg = (
                "Provider not set in configuration - this should not happen "
                "with proper config flow"
            )
            provider_error = "Provider configuration is required"
            LOGGER.error(provider_required_msg)
            raise ValueError(provider_error)

        if not self.group:
            group_required_msg = (
                "Group not set in configuration - this should not happen "
                "with proper config flow"
            )
            group_error = "Group configuration is required"
            LOGGER.error(group_required_msg)
            raise ValueError(group_error)

        # Initialize with names first, then we'll update with IDs when we fetch data
        self.region_id = None
        self.provider_id = None
        self._provider_name = ""  # Cache the provider name

        # Use the provided API instance
        self.api = api
        # Note: We'll resolve IDs and update API during first data update

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

        output = CalendarEvent(
            summary=summary,
            start=event.start,
            end=event.end,
            description=event_type,
            uid=event_type,
        )
        LOGGER.debug("Calendar Event: %s", output)
        return output

    def _event_to_state(self, event: CalendarEvent | None) -> str | None:
        if not event:
            return STATE_NORMAL

        # Map event types to states using uid field
        if event.uid == OutageEventType.DEFINITE.value:
            return STATE_OUTAGE

        LOGGER.warning("Unknown event type: %s", event.uid)
        return STATE_NORMAL

    def _simplify_provider_name(self, provider_name: str) -> str:
        """Simplify provider names for cleaner display in device names."""
        # Replace long DTEK provider names with just "ДТЕК"
        if PROVIDER_DTEK_FULL in provider_name.upper():
            return PROVIDER_DTEK_SHORT

        # Add more provider simplifications here as needed
        return provider_name

    async def async_sync_events_to_calendar(self) -> None:
        """Sync events to external calendar if configured."""
        calendar_entity = self.config_entry.options.get(
            CONF_CALENDAR,
            self.config_entry.data.get(CONF_CALENDAR),
        )

        if not calendar_entity:
            LOGGER.debug("No calendar entity configured for sync")
            return

        try:
            LOGGER.debug("Syncing events to calendar: %s", calendar_entity)

            # Get events for today and tomorrow + a week ahead
            now = dt_utils.now()
            end_date = now + datetime.timedelta(days=8)
            events = self.get_events_between(now, end_date)

            # Get the calendar service
            if "calendar" not in self.hass.services.async_services():
                LOGGER.warning("Calendar service not available")
                return

            # Get existing events from the calendar
            try:
                existing_events = await self._get_calendar_events(
                    calendar_entity, now, end_date
                )
            except Exception:  # noqa: BLE001
                LOGGER.debug("Could not retrieve existing calendar events")
                existing_events = []

            # Create events in the calendar (skip duplicates)
            for event in events:
                # Check if event already exists
                if not self._event_exists(event, existing_events):
                    await self._create_calendar_event(calendar_entity, event)
                else:
                    LOGGER.debug("Event already exists, skipping: %s", event.summary)

        except Exception:
            LOGGER.exception("Failed to sync events to calendar")

    async def _get_calendar_events(
        self,
        calendar_entity: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
    ) -> list[dict]:
        """Get events from the external calendar."""
        try:
            # Use calendar.get_events service to fetch existing events
            response = await self.hass.services.async_call(
                "calendar",
                "get_events",
                {
                    "entity_id": calendar_entity,
                    "duration": {
                        "days": (end_date - start_date).days + 1,
                    },
                },
                return_response=True,
            )
            events = response.get(calendar_entity, {}).get("events", [])
            LOGGER.debug("Retrieved %d existing events from calendar", len(events))
        except Exception:  # noqa: BLE001
            LOGGER.debug("Could not fetch events from calendar")
            return []
        else:
            return events

    def _event_exists(self, event: CalendarEvent, existing_events: list[dict]) -> bool:
        """Check if event already exists in the calendar."""
        # Match by start time and summary (allows for slight time variations)
        for existing in existing_events:
            existing_start = existing.get("start")
            existing_summary = existing.get("summary", "")

            if existing_start and existing_summary == event.summary:
                # Compare times (allow tolerance for time differences)
                try:
                    if isinstance(existing_start, str):
                        existing_dt = dt_utils.parse_datetime(existing_start)
                    else:
                        existing_dt = existing_start

                    if (
                        existing_dt
                        and abs((existing_dt - event.start).total_seconds())
                        < CALENDAR_SYNC_TIME_TOLERANCE
                    ):
                        return True
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Could not compare event times")
                    continue

        return False

    async def _create_calendar_event(
        self, calendar_entity: str, event: CalendarEvent
    ) -> None:
        """Create an event in the external calendar."""
        try:
            # Use calendar service to create event
            await self.hass.services.async_call(
                "calendar",
                "create_event",
                {
                    "entity_id": calendar_entity,
                    "summary": event.summary,
                    "description": event.description,
                    "start_date_time": event.start.isoformat(),
                    "end_date_time": event.end.isoformat(),
                },
            )
            LOGGER.debug("Created event in calendar: %s", event.summary)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Could not create event")
