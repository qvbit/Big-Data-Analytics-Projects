import sys

# Script reads lines from standard input and with some simple processing outputs to standard output event_name: 1 as key-value pairs.

for line in sys.stdin:
    line = line.strip()
    splits = line.split(',')

    if len(splits) != 4:
        # Ignore problematic lines
        continue

    # Unpack all of the attributes
    patient_id, event_name, date_offset, value = splits

    # Emit key-value pair event_name: 1
    print(event_name + '\\' + '1')






