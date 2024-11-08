class Box:
    def __init__(self, port, weight):
        self.port = port       # Destination port
        self.weight = weight   # Weight of the box

def min_trips(boxes, ports_count, max_boxes, max_weight):
    trips = []
    current_trip = []

    for box in boxes:
        if len(current_trip) + 1 <= max_boxes and sum(box.weight for box in current_trip) + box.weight <= max_weight:
            current_trip.append(box)
        else:
            trips.append(current_trip)
            current_trip = [box]

    if current_trip:
        trips.append(current_trip)

    return trips

def main():
    boxes = [Box(1, 1), Box(1, 2), Box(2, 1), Box(2, 2), Box(3, 3)]
    ports_count = 3
    max_boxes = 5
    max_weight = 6

    trips = min_trips(boxes, ports_count, max_boxes, max_weight)

    port_trips = {port: [] for port in range(1, ports_count + 1)}
    for i, trip in enumerate(trips):
        for box in trip:
            port = box.port
            port_trips[port].append(box.weight)

    print(f"Minimum number of trips: {len(trips) + 1}")
    for port in range(1, ports_count + 1):
        if port_trips[port]:
            print(f"Trip {port} (Port {port}): {port_trips[port]}")

if __name__ == "__main__":
    main()
