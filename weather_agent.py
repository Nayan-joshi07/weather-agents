from __future__ import annotations as _annotations
import asyncio
import os
from dataclasses import dataclass
from typing import Any
import logfire
from devtools import debug
from httpx import AsyncClient, HTTPStatusError, ReadError
from pydantic_ai import Agent, ModelRetry, RunContext


# Set OpenAI API key - REPLACE THIS WITH YOUR ACTUAL KEY
os.environ["OPENAI_API_KEY"] = "sk-ktx6SIrnyDM0PnIZr4b_CnOawNqo2P8jwru1VODBKBT3BlbkFJRHrkec-Q0IcFWaS0vSVuBJ7iVwN_9HUvfnbXie4n0A"

logfire.configure(send_to_logfire='if-token-present')


@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None


weather_agent = Agent(
    'openai:gpt-4o',
    system_prompt=(
        'Be concise, reply with one sentence.'
        'Use the `get_lat_lng` tool to get the latitude and longitude of the locations,'
        'then use the `get_weather` tool to get the weather.'
    ),
    deps_type=Deps,
    retries=2,
    instrument=True,
)


@weather_agent.tool
async def get_lat_lng(
        ctx: RunContext[Deps], location_description: str) -> dict[str, float]:
    """Get the latitude and longitude of a location.
    Args:
    ctx : The context.
    location_description:A Description of a location.
    """
    # Fallback locations for common places to avoid API calls
    fallbacks = {
        "london": {"lat": 51.5074, "lng": -0.1278},
        "wiltshire": {"lat": 51.0632, "lng": -1.9497},
        # Add more common locations if needed
    }

    # Check if we have a fallback for this location
    location_key = location_description.lower().strip()
    if location_key in fallbacks:
        print(f"Using fallback coordinates for {location_description}")
        return fallbacks[location_key]

    # If no API key or we want to avoid API rate limits, use fallbacks broadly
    if ctx.deps.geo_api_key is None or os.getenv("USE_FALLBACKS") == "1":
        print(
            f"No API key or using fallbacks, returning dummy location for {location_description}")
        return {"lat": 51.1, "lng": -0.1}  # Default fallback

    if ctx.deps.geo_api_key is None:
        # if no API key is provided , return a dummy response(London)
        return {'lat': 51.1, 'lng': -0.1}
    params = {
        'q': location_description,
        'api_key': ctx.deps.geo_api_key,
    }

    try:
        with logfire.span('calling geocode API', params=params) as span:
            r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
            r.raise_for_status()
            data = r.json()
            span.set_attribute('response', data)

        if data:
            return {'lat': float(data[0]['lat']), 'lng': float(data[0]['lon'])}
        else:
            print(
                f"No results found for {location_description}, using fallback")
            return {"lat": 51.1, "lng": -0.1}  # Default fallback

    except (HTTPStatusError, ReadError, Exception) as e:
        print(f"Error in geocoding API: {str(e)}, using fallback coordinates")
        return {"lat": 51.1, "lng": -0.1}  # Default fallback


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.weather_api_key is None:
        # if no API key is provided, return a dummy response
        return {'temperature': '21 °C', 'description': 'Sunny'}

    params = {
        'apikey': ctx.deps.weather_api_key,
        'location': f'{lat},{lng}',
        'units': 'metric',
    }
    with logfire.span('calling weather API', params=params) as span:
        r = await ctx.deps.client.get(
            'https://api.tomorrow.io/v4/weather/realtime', params=params
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)

    values = data['data']['values']
    # https://docs.tomorrow.io/reference/data-layers-weather-codes
    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }
    return {
        'temperature': f'{values["temperatureApparent"]:0.0f}°C',
        'description': code_lookup.get(values['weatherCode'], 'Unknown'),
    }


async def main():
    async with AsyncClient() as client:
        weather_api_key = os.getenv('WEATHER_API_KEY')
        geo_api_key = os.getenv('GEO_API_KEY')
        deps = Deps(
            client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key
        )
        os.environ["USE_FALLBACKS"] = "1"
        result = await weather_agent.run(
            'What is the weather like in London?', deps=deps
        )
        debug(result)
        print('Response:', result.data)

if __name__ == '__main__':
    asyncio.run(main())
