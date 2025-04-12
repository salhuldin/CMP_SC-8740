def parse_fasta(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    seq_id = None
    sequences = {}
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            seq_id = line[1:]
            sequences[seq_id] = ''
        else:
            sequences[seq_id] += line
    return sequences

def parse_label_fasta(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    seq_id = None
    labels = {}
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            seq_id = line[1:]
            labels[seq_id] = ''
        else:
            labels[seq_id] += line
    return {seq_id: list(map(int, lab.split(','))) for seq_id, lab in labels.items()}
