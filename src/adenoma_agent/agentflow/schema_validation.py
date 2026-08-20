import json
from pathlib import Path


class ReviewerSchemaValidationError(ValueError):
    pass


class ReviewerSchemaDependencyError(RuntimeError):
    pass


class ReviewerJsonSchemaValidator(object):
    """Draft 2020-12 validator for the public Reviewer wire contracts.

    The dependency is optional for offline control-flow tests, but production
    entry points can set ``require_dependency=True`` and fail explicitly when
    the contract extra has not been installed.
    """

    FILES = {
        "common": "reviewer_common_v1.schema.json",
        "request": "reviewer_task_request_v1.schema.json",
        "observation": "reviewer_observation_v1.schema.json",
        "ledger": "reviewer_ledger_record_v1.schema.json",
    }

    def __init__(self, schema_dir=None, require_dependency=False):
        self.schema_dir = Path(schema_dir) if schema_dir else self._default_schema_dir()
        self.available = False
        self._validators = {}
        try:
            from jsonschema import Draft202012Validator, FormatChecker
        except (ImportError, AttributeError) as exc:
            if require_dependency:
                raise ReviewerSchemaDependencyError(
                    "Reviewer runtime requires jsonschema>=4.18; install the contracts extra"
                ) from exc
            return
        schemas = {}
        for contract_name, filename in self.FILES.items():
            path = self.schema_dir / filename
            with path.open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
            schema["$id"] = path.resolve().as_uri()
            Draft202012Validator.check_schema(schema)
            schemas[contract_name] = schema
        try:
            from referencing import Registry, Resource
        except ImportError:
            Registry = None
            Resource = None
        if Registry is not None:
            resources = [
                (schema["$id"], Resource.from_contents(schema))
                for schema in schemas.values()
            ]
            resource_registry = Registry().with_resources(resources)
            for contract_name, schema in schemas.items():
                if contract_name == "common":
                    continue
                self._validators[contract_name] = Draft202012Validator(
                    schema,
                    registry=resource_registry,
                    format_checker=FormatChecker(),
                )
        else:
            try:
                from jsonschema import RefResolver
            except ImportError as exc:
                if require_dependency:
                    raise ReviewerSchemaDependencyError(
                        "The installed jsonschema package lacks a compatible reference resolver"
                    ) from exc
                return
            store = {}
            for contract_name, schema in schemas.items():
                store[schema["$id"]] = schema
                store[self.FILES[contract_name]] = schema
            for contract_name, schema in schemas.items():
                if contract_name == "common":
                    continue
                self._validators[contract_name] = Draft202012Validator(
                    schema,
                    resolver=RefResolver.from_schema(schema, store=store),
                    format_checker=FormatChecker(),
                )
        self.available = True

    def validate_request(self, payload):
        self._validate("request", payload)

    def validate_observation(self, payload):
        self._validate("observation", payload)

    def validate_ledger_record(self, payload):
        self._validate("ledger", payload)

    def _validate(self, contract_name, payload):
        if not self.available:
            return
        errors = sorted(
            self._validators[contract_name].iter_errors(payload),
            key=lambda error: tuple(str(item) for item in error.absolute_path),
        )
        if not errors:
            return
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise ReviewerSchemaValidationError(
            "{0} contract violation at {1}: {2}".format(contract_name, location, error.message)
        )

    @staticmethod
    def _default_schema_dir():
        return Path(__file__).resolve().parents[3] / "docs" / "model" / "contracts" / "schemas"
