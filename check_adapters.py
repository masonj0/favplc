import fortuna
import fortuna_analytics

def check_results_adapters():
    classes = fortuna_analytics.get_results_adapter_classes()
    print(f"Found {len(classes)} results adapter classes:")
    for cls in classes:
        print(f" - {cls.__name__} (SOURCE_NAME: {getattr(cls, 'SOURCE_NAME', 'N/A')})")

if __name__ == "__main__":
    check_results_adapters()
