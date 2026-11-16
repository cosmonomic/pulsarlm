import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import CallbackOptions, Histogram, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.types import AttributeValue
from prometheus_client import CollectorRegistry

from pulsarlm.completion import CompletionUsage
from pulsarlm.core.engine import EngineStat

PROVIDER_NAME = "pulsarlm"
REQUEST_DURATION = "gen_ai.server.request.duration"
TIME_TO_FIRST_TOKEN = "gen_ai.server.time_to_first_token"
TIME_PER_OUTPUT_TOKEN = "gen_ai.server.time_per_output_token"
PEAK_DEVICE_MEMORY = "pulsarlm.usage.peak_device_memory"
INPUT_TOKENS = "pulsarlm.usage.input_tokens"
OUTPUT_TOKENS = "pulsarlm.usage.output_tokens"
INPUT_TOKENS_ON_DEVICE = "pulsarlm.usage.input_tokens_on_device"
INPUT_TOKENS_IN_STORE = "pulsarlm.usage.input_tokens_in_store"
# The GenAI semantic conventions' bounds for token histograms.
TOKEN_BUCKETS = tuple(float(4**exponent) for exponent in range(14))
BUCKET_BOUNDARIES = {
    REQUEST_DURATION: (
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
        40.96,
        81.92,
    ),
    TIME_TO_FIRST_TOKEN: (
        0.001,
        0.005,
        0.01,
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    ),
    TIME_PER_OUTPUT_TOKEN: (
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.75,
        1.0,
        2.5,
    ),
    PEAK_DEVICE_MEMORY: tuple(float(2**exponent) for exponent in range(20, 41)),
    INPUT_TOKENS: TOKEN_BUCKETS,
    OUTPUT_TOKENS: TOKEN_BUCKETS,
    INPUT_TOKENS_ON_DEVICE: TOKEN_BUCKETS,
    INPUT_TOKENS_IN_STORE: TOKEN_BUCKETS,
}


@dataclass(frozen=True)
class Telemetry:
    """The GenAI model server metrics of the OpenTelemetry semantic conventions,
    and pulsarlm's own under pulsarlm.*. registry serves them in the Prometheus
    text format; they are also pushed over OTLP when the OTEL_EXPORTER_OTLP_ENDPOINT
    or OTEL_EXPORTER_OTLP_METRICS_ENDPOINT environment variable is set. tracer's
    spans are pushed over OTLP when OTEL_EXPORTER_OTLP_ENDPOINT or
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is set."""

    registry: CollectorRegistry
    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    tracer: Tracer
    request_duration: Histogram
    time_to_first_token: Histogram
    time_per_output_token: Histogram
    peak_device_memory: Histogram
    input_tokens: Histogram
    output_tokens: Histogram
    input_tokens_on_device: Histogram
    input_tokens_in_store: Histogram

    def shutdown(self) -> None:
        """Exports what the providers hold, then stops them."""
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()


def exports_over_otlp(signal_endpoint: str) -> bool:
    return bool({"OTEL_EXPORTER_OTLP_ENDPOINT", signal_endpoint} & set(os.environ))


def start_telemetry(stat: Callable[[], EngineStat], model: str) -> Telemetry:
    resource = Resource.create({"service.name": PROVIDER_NAME})
    tracer_provider = TracerProvider(resource=resource, shutdown_on_exit=False)
    if exports_over_otlp("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    registry = CollectorRegistry()
    readers: list[MetricReader] = [PrometheusMetricReader(registry=registry)]
    if exports_over_otlp("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    meter_provider = MeterProvider(
        shutdown_on_exit=False,
        metric_readers=readers,
        resource=resource,
        views=[
            View(
                instrument_name=name,
                aggregation=ExplicitBucketHistogramAggregation(boundaries),
            )
            for name, boundaries in BUCKET_BOUNDARIES.items()
        ],
    )
    meter = meter_provider.get_meter(PROVIDER_NAME)

    def device_memory(_: CallbackOptions) -> Iterable[Observation]:
        yield Observation(
            stat().kv_cache_bytes,
            {"gen_ai.provider.name": PROVIDER_NAME, "gen_ai.response.model": model},
        )

    meter.create_observable_gauge(
        "pulsarlm.device_memory",
        callbacks=[device_memory],
        unit="By",
        description="Bytes of the KV cache that running requests use.",
    )
    return Telemetry(
        registry=registry,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        tracer=tracer_provider.get_tracer(PROVIDER_NAME),
        request_duration=meter.create_histogram(REQUEST_DURATION, unit="s"),
        time_to_first_token=meter.create_histogram(TIME_TO_FIRST_TOKEN, unit="s"),
        time_per_output_token=meter.create_histogram(TIME_PER_OUTPUT_TOKEN, unit="s"),
        peak_device_memory=meter.create_histogram(
            PEAK_DEVICE_MEMORY,
            unit="By",
            description="The most bytes of each device's KV cache pool a request's "
            "lines took at once, lines shared with other requests included.",
        ),
        input_tokens=meter.create_histogram(INPUT_TOKENS, unit="{token}"),
        output_tokens=meter.create_histogram(OUTPUT_TOKENS, unit="{token}"),
        input_tokens_on_device=meter.create_histogram(
            INPUT_TOKENS_ON_DEVICE,
            unit="{token}",
            description="Prompt tokens of the resumed state still in the KV cache.",
        ),
        input_tokens_in_store=meter.create_histogram(
            INPUT_TOKENS_IN_STORE,
            unit="{token}",
            description="Prompt tokens of the resumed state fetched from the store.",
        ),
    )


class RequestTelemetry:
    """Records one request: the metrics when it ends, and its request span when it
    finishes. The span carries the metrics' attributes and request_attributes."""

    def __init__(
        self,
        telemetry: Telemetry,
        request_model: str,
        response_model: str,
        request_attributes: dict[str, AttributeValue],
    ) -> None:
        self._started_ns = time.time_ns()
        self._started_at = time.monotonic()
        self._telemetry = telemetry
        self._metric_attributes: dict[str, AttributeValue] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": PROVIDER_NAME,
            "gen_ai.request.model": request_model,
            "gen_ai.response.model": response_model,
        }
        self._request_attributes = request_attributes

    def finish(self, usage: CompletionUsage, finish_reason: str) -> None:
        duration = time.monotonic() - self._started_at
        telemetry = self._telemetry
        attributes = self._metric_attributes
        store_tokens = usage.cached_tokens - usage.device_cached_tokens
        telemetry.request_duration.record(duration, attributes)
        if usage.time_to_first_token_s is not None:
            telemetry.time_to_first_token.record(
                usage.time_to_first_token_s, attributes
            )
        if usage.completion_tokens > 1:
            telemetry.time_per_output_token.record(
                usage.decode_time_s / (usage.completion_tokens - 1), attributes
            )
        telemetry.peak_device_memory.record(usage.peak_device_memory, attributes)
        telemetry.input_tokens.record(usage.prompt_tokens, attributes)
        telemetry.output_tokens.record(usage.completion_tokens, attributes)
        telemetry.input_tokens_on_device.record(usage.device_cached_tokens, attributes)
        telemetry.input_tokens_in_store.record(store_tokens, attributes)
        span_attributes: dict[str, AttributeValue] = (
            attributes
            | self._request_attributes
            | {
                "gen_ai.response.finish_reasons": (finish_reason,),
                "gen_ai.usage.input_tokens": usage.prompt_tokens,
                "gen_ai.usage.output_tokens": usage.completion_tokens,
                "pulsarlm.request.duration": duration,
                "pulsarlm.decode.duration": usage.decode_time_s,
                "pulsarlm.usage.input_tokens_on_device": usage.device_cached_tokens,
                "pulsarlm.usage.input_tokens_in_store": store_tokens,
                "pulsarlm.usage.peak_device_memory": usage.peak_device_memory,
            }
        )
        if usage.time_to_first_token_s is not None:
            span_attributes["pulsarlm.time_to_first_token"] = (
                usage.time_to_first_token_s
            )
        telemetry.tracer.start_span(
            "request",
            kind=SpanKind.SERVER,
            attributes=span_attributes,
            start_time=self._started_ns,
        ).end()

    def fail(self, error_type: str) -> None:
        self._telemetry.request_duration.record(
            time.monotonic() - self._started_at,
            self._metric_attributes | {"error.type": error_type},
        )
