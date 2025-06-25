import os
import subprocess
import re
import magic

BENIGN_ROOT = "IoTMalwareDetection-master/Benign/all_goodware/"
MALWARE_ROOT = "IoTMalwareDetection-master/Malware(Disassembeled)/"


# DETECTS MULTIPLE OPCODES PER LINE
opcode_regex = re.compile(r'^\s*[0-9a-fA-F]+:\s+[0-9a-fA-F\s]+\s+(.+)')
instruction_regex = re.compile(r'\b([a-zA-Z][a-zA-Z0-9.]*)\b')


def process_directory(root_dir, is_disassembled=False):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if is_disassembled:
                # For already-disassembled files (.asm)
                if not file.endswith(".asm"):
                    continue
                asm_path = file_path
                opcode_path = file_path.replace(".asm", ".opcode")
            else:
                # For raw binaries
                asm_path = file_path + ".asm"
                opcode_path = file_path + ".opcode"
                # Validate file type using libmagic
                try:
                    file_type = magic.from_file(file_path)
                    if "ELF 32-bit" not in file_type or "ARM" not in file_type:
                        print(f"Skipping non-ARM ELF file: {file_path}")
                        continue
                except Exception as e:
                    print(f"Could not determine file type for {file_path}: {e}")
                    continue
                # Disassemble with ARM objdump
                try:
                    with open(asm_path, "w") as asm_file:
                        result = subprocess.run(
                            ["arm-linux-gnueabi-objdump", "-d", file_path],
                            stdout=asm_file, stderr=subprocess.PIPE, text=True)
                        if result.returncode != 0 or os.path.getsize(asm_path) == 0:
                            print(f"Failed to disassemble {file_path}: {result.stderr.strip()}")
                            continue
                except Exception as e:
                    print(f"Exception while disassembling {file_path}: {e}")
                    continue

            # Extract opcodes
            try:
                extracted = False
                with open(asm_path, "r") as asm_file, open(opcode_path, "w") as opcode_file:
                    for line in asm_file:
                        match = opcode_regex.match(line)
                        if match:
                            instruction_part = match.group(1)
                            # Find all opcodes in the instruction part
                            opcodes = instruction_regex.findall(instruction_part)
                            for opcode in opcodes:
                                # Skip register names and immediate values
                                if not opcode.lower().startswith(('r', 'sp', 'lr', 'pc', 'fp', 'ip')) and not opcode.isdigit():
                                    opcode_file.write(opcode + '\n')
                                    extracted = True
                if not extracted:
                    print(f"No opcodes found in {file_path}")
            except Exception as e:
                print(f"Failed to extract opcodes from {asm_path}: {e}")
                continue

# Process benign binaries (raw ELF files)
process_directory(BENIGN_ROOT, is_disassembled=False)
# Process malware disassemblies (.asm files)
process_directory(MALWARE_ROOT, is_disassembled=True)