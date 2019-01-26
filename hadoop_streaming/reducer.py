import sys

current_event = None
current_count = 0
event_name = None

for line in sys.stdin:

    line = line.strip()

    event_name, count = line.split('\t', 1)

    try:
        count = int(count)
    except ValueError:
        # Count was not a number so just discard this line
        continue

    # Line is sorted by event_name
    if current_event == event_name:
        # Accumulate on same key
        current_count += 1
    else:
        # New key
        if current_event:
            # Write result to stdout. Since events are sorted, we are done with counting the frequency of this specific event.
            print('%s\t%s' % (current_event, current_count))
        current_count = count
        current_event = event_name

# Remember to output the last event_name.
if current_event == event_name:
    print('%s\t%s' % (current_event, current_count))



