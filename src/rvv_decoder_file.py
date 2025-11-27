#!/usr/bin/env python3
"""
RISC-V Vector (RVV) Instruction Decoder
"""

import re
import sys
import argparse

class RVVDecoder:
    def __init__(self, debug=False):
        self.debug = debug
        
        # Field masks
        self.OPCODE_MASK = 0x7F
        self.FUNCT3_MASK = 0x7000
        self.FUNCT6_MASK = 0xFC000000
        self.RD_MASK = 0xF80
        self.RS1_MASK = 0xF8000
        self.RS2_MASK = 0x1F00000
        self.VS2_MASK = 0x1F00000
        self.VS1_MASK = 0xF8000
        self.VD_MASK = 0xF80
        self.VM_MASK = 0x2000000
        self.WIDTH_MASK = 0x7000
        self.MOP_MASK = 0x0C000000
        self.NF_MASK = 0xE0000000
        self.IMM_MASK = 0xF8000
        self.LUMOP_MASK = 0x1F00000
        self.SUMOP_MASK = 0x1F00000
        self.ZIMM_MASK = 0x7FF00000
        
        # Opcodes
        self.OP_V = 0x57
        self.LOAD_FP = 0x07
        self.STORE_FP = 0x27
        
        # Funct3 types
        self.OPIVV = 0b000  
        self.OPFVV = 0b001  
        self.OPMVV = 0b010  
        self.OPIVI = 0b011 
        self.OPIVX = 0b100 
        self.OPFVF = 0b101  
        self.OPMVX = 0b110  
        self.OPCFG = 0b111 
        
        # Configuration maps
        self.width_map = {
            0b000: "8", 
            0b101: "16", 
            0b110: "32", 
            0b111: "64",
            0b001: "FP16", 
            0b010: "FP32", 
            0b011: "FP64", 
            0b100: "FP128"
        }
        
        self.vsew_map = {
            0b000: "e8", 
            0b001: "e16", 
            0b010: "e32", 
            0b011: "e64"
        }
        
        self.vlmul_map = {
            0b000: "m1", 
            0b001: "m2", 
            0b010: "m4", 
            0b011: "m8",
            0b101: "mf8", 
            0b110: "mf4", 
            0b111: "mf2"
        }
        
        self.whole_reg_moves = {
            0b00000: 0, 
            0b00001: 1, 
            0b00011: 2, 
            0b00111: 3
        }
        
        self.nfield_map = {
            0b000: 1, 
            0b001: 2, 
            0b010: 3, 
            0b011: 4, 
            0b100: 5, 
            0b101: 6, 
            0b110: 7, 
            0b111: 8
        }
        
        self.setup_instruction_tables()

    def debug_print(self, message):
        """Conditional debug print"""
        if self.debug:
            print(f"[DEBUG] {message}")

    def setup_instruction_tables(self):
        """Setup instruction tables based on the provided documentation"""
        
        # === OPI Instructions (Integer) ===
        self.opi_instructions = {
            # Basic arithmetic
            0b000000: "vadd",
            0b000010: "vsub", 
            0b000011: "vrsub",
            # Min/max
            0b000100: "vminu",
            0b000101: "vmin",
            0b000110: "vmaxu",
            0b000111: "vmax",
            # Logical
            0b001001: "vand",
            0b001010: "vor", 
            0b001011: "vxor",
            # Gather/slide
            0b001100: "vrgather",
            0b001110: "vslideup",
            0b001111: "vslidedown",
            # Carry/mask
            0b010000: "vadc",
            0b010001: "vmadc", 
            0b010010: "vsbc",
            0b010011: "vmsbc",
            0b010111: "vmerge",
            # Comparison
            0b011000: "vmseq",
            0b011001: "vmsne",
            0b011010: "vmsltu",
            0b011011: "vmslt",
            0b011100: "vmsleu", 
            0b011101: "vmsle",
            0b011110: "vmsgtu",
            0b011111: "vmsgt",
            # Saturating arithmetic
            0b100000: "vsaddu",
            0b100001: "vsadd",
            0b100010: "vssubu",
            0b100011: "vssub",
            # Shift
            0b100101: "vsll",
            0b101000: "vsrl", 
            0b101001: "vsra",
            # Narrowing shift
            0b101010: "vssrl",
            0b101011: "vssra",
            0b101100: "vnsrl",
            0b101101: "vnsra",
            0b101110: "vnclipu",
            0b101111: "vnclip",
            # Multiply
            0b100111: "vsmul",
            # Reduction
            0b110000: "vwredsumu",
            0b110001: "vwredsum"
        }

        # === OPM Instructions (Mask/Multiply) ===
        self.opm_instructions = {
            # Reduction
            0b000000: "vredsum",
            0b000001: "vredand",
            0b000010: "vredor", 
            0b000011: "vredxor",
            0b000100: "vredminu",
            0b000101: "vredmin",
            0b000110: "vredmaxu",
            0b000111: "vredmax",
            # Fixed-point
            0b001000: "vaaddu",
            0b001001: "vaadd",
            0b001010: "vasubu", 
            0b001011: "vasub",
            # Special unary
            0b010000: "VWXUNARY0",
            0b010010: "VXUNARY0",
            0b010100: "VMUNARY0",
            # Mask
            0b010111: "vcompress",
            0b011000: "vmandn",
            0b011001: "vmand",
            0b011010: "vmor",
            0b011011: "vmxor",
            0b011100: "vmorn",
            0b011101: "vmnand", 
            0b011110: "vmnor",
            0b011111: "vmxnor",
            # Divide/remainder
            0b100000: "vdivu",
            0b100001: "vdiv",
            0b100010: "vremu",
            0b100011: "vrem",
            # Multiply
            0b100100: "vmulhu",
            0b100101: "vmul",
            0b100110: "vmulhsu",
            0b100111: "vmulh",
            # Multiply-add
            0b101001: "vmadd",
            0b101011: "vnmsub",
            0b101101: "vmacc",
            0b101111: "vnmsac",
            # Widening
            0b110000: "vwaddu",
            0b110001: "vwadd",
            0b110010: "vwsubu",
            0b110011: "vwsub",
            0b110100: "vwaddu.w",
            0b110101: "vwadd.w",
            0b110110: "vwsubu.w", 
            0b110111: "vwsub.w",
            0b111000: "vwmulu",
            0b111010: "vwmulsu",
            0b111011: "vwmul",
            0b111100: "vwmaccu",
            0b111101: "vwmacc",
            0b111110: "vwmaccus",
            0b111111: "vwmaccsu",
            # Slide
            0b001110: "vslide1up",
            0b001111: "vslide1down"
        }

        # === OPF Instructions (Floating-Point) ===
        self.opf_instructions = {
            # Basic arithmetic
            0b000000: "vfadd",
            0b000010: "vfsub",
            0b000100: "vfmin", 
            0b000110: "vfmax",
            # Reduction
            0b000001: "vfredusum",
            0b000011: "vfredosum",
            0b000101: "vfredmin",
            0b000111: "vfredmax",
            # Sign injection
            0b001000: "vfsgnj",
            0b001001: "vfsgnjn",
            0b001010: "vfsgnjx",
            # Slide
            0b001110: "vfslide1up",
            0b001111: "vfslide1down",
            # Special unary
            0b010000: "VWFUNARY0",
            0b010010: "VFUNARY0", 
            0b010011: "VFUNARY1",
            # Merge/move
            0b010111: "vfmerge",
            # Comparison
            0b011000: "vmfeq",
            0b011001: "vmfle",
            0b011011: "vmflt",
            0b011100: "vmfne",
            0b011101: "vmfgt", 
            0b011111: "vmfge",
            # Divide
            0b100000: "vfdiv",
            0b100001: "vfrdiv",
            # Multiply
            0b100100: "vfmul",
            0b100111: "vfrsub",
            # Fused multiply-add
            0b101000: "vfmadd",
            0b101001: "vfnmadd",
            0b101010: "vfmsub",
            0b101011: "vfnmsub",
            0b101100: "vfmacc",
            0b101101: "vfnmacc",
            0b101110: "vfmsac",
            0b101111: "vfnmsac",
            # Widening
            0b110000: "vfwadd",
            0b110010: "vfwsub",
            0b110100: "vfwadd.w",
            0b110110: "vfwsub.w",
            0b111000: "vfwmul",
            0b111100: "vfwmacc",
            0b111101: "vfwnmacc",
            0b111110: "vfwmsac",
            0b111111: "vfwnmsac",
            # Widening reduction
            0b110001: "vfwredusum",
            0b110011: "vfwredosum",
        }

        # Special operation mappings
        self.vwxunary0_map = {
            0b00000: "vmv.x.s",
            0b10000: "vcpop", 
            0b10001: "vfirst"
        }
        
        self.vrxunary0_map = {
            0b00000: "vmv.s.x"
        }
        
        self.vxunary0_map = {
            0b00010: "vzext.vf8",
            0b00011: "vsext.vf8", 
            0b00100: "vzext.vf4",
            0b00101: "vsext.vf4",
            0b00110: "vzext.vf2",
            0b00111: "vsext.vf2"
        }
        
        self.vrfunary0_map = {
            0b00000: "vfmv.s.f"
        }
        
        self.vwfunary0_map = {
            0b00000: "vfmv.f.s"
        }
        
        self.vfunary0_map = {
            # Single-width converts
            0b00000: "vfcvt.xu.f.v",
            0b00001: "vfcvt.x.f.v",
            0b00010: "vfcvt.f.xu.v", 
            0b00011: "vfcvt.f.x.v",
            0b00110: "vfcvt.rtz.xu.f.v",
            0b00111: "vfcvt.rtz.x.f.v",
            # Widening converts
            0b01000: "vfwcvt.xu.f.v",
            0b01001: "vfwcvt.x.f.v",
            0b01010: "vfwcvt.f.xu.v",
            0b01011: "vfwcvt.f.x.v", 
            0b01100: "vfwcvt.f.f.v",
            0b01110: "vfwcvt.rtz.xu.f.v",
            0b01111: "vfwcvt.rtz.x.f.v",
            # Narrowing converts
            0b10000: "vfncvt.xu.f.w",
            0b10001: "vfncvt.x.f.w",
            0b10010: "vfncvt.f.xu.w",
            0b10011: "vfncvt.f.x.w",
            0b10100: "vfncvt.f.f.w",
            0b10101: "vfncvt.rod.f.f.w",
            0b10110: "vfncvt.rtz.xu.f.w",
            0b10111: "vfncvt.rtz.x.f.w"
        }
        
        self.vfunary1_map = {
            0b00000: "vfsqrt.v",
            0b00100: "vfrsqrt7.v",
            0b00101: "vfrec7.v",
            0b10000: "vfclass.v"
        }
        
        self.vmunary0_map = {
            0b00001: "vmsbf",
            0b00010: "vmsof", 
            0b00011: "vmsif",
            0b10000: "viota",
            0b10001: "vid"
        }

    def extract_fields(self, instruction):
        """Extract all relevant fields from a 32-bit instruction"""
        fields = {
            'opcode': instruction & self.OPCODE_MASK,
            'funct3': (instruction & self.FUNCT3_MASK) >> 12,
            'funct6': (instruction & self.FUNCT6_MASK) >> 26,
            'rd'    : (instruction & self.RD_MASK) >> 7,
            'rs1'   : (instruction & self.RS1_MASK) >> 15,
            'rs2'   : (instruction & self.RS2_MASK) >> 20,
            'vs2'   : (instruction & self.VS2_MASK) >> 20,
            'vs1'   : (instruction & self.VS1_MASK) >> 15,
            'vd'    : (instruction & self.VD_MASK) >> 7,
            'vm'    : (instruction & self.VM_MASK) >> 25,
            'width' : (instruction & self.WIDTH_MASK) >> 12,
            'mop'   : (instruction & self.MOP_MASK) >> 26,
            'nf'    : (instruction & self.NF_MASK) >> 29,
            'imm'   : (instruction & self.IMM_MASK) >> 15,
            'lumop' : (instruction & self.LUMOP_MASK) >> 20,
            'sumop' : (instruction & self.SUMOP_MASK) >> 20,
            'zimm'  : (instruction & self.ZIMM_MASK) >> 20,
            'raw'   : instruction
        }
        
        if self.debug:
            self.debug_print("=== FIELD EXTRACTION ===")
            self.debug_print(f"Raw instruction: 0x{instruction:08x}")
            self.debug_print(f"Opcode: 0x{fields['opcode']:02x}")
            self.debug_print(f"Funct3: {fields['funct3']:03b}")
            self.debug_print(f"Funct6: {fields['funct6']:06b}")
            
        return fields

    def decode_vtype(self, zimm):
        """Decode vtype configuration from zimm field"""
        vlmul = zimm & 0b111
        vsew  = (zimm >> 3) & 0b111
        vta   = (zimm >> 6) & 1
        vma   = (zimm >> 7) & 1
        vill  = (zimm >> 8) & 1
        
        parts = []
        if vill:
            parts.append("vill=1")
        else:
            if vsew in self.vsew_map:
                parts.append(self.vsew_map[vsew])
            if vlmul in self.vlmul_map:
                parts.append(self.vlmul_map[vlmul])
            parts.append("ta" if vta else "tu")
            parts.append("ma" if vma else "mu")
        return ", ".join(parts)

    def decode_config_instruction(self, fields):
        """Decode vector configuration instructions"""
        if fields['funct3'] != self.OPCFG:
            return None
        rd   = fields['rd']
        rs1  = fields['rs1']
        rs2  = fields['rs2']
        zimm = fields['zimm']
        
        if self.debug:
            self.debug_print(f"{fields['raw'] >> 31}")
            
        if (fields['raw'] >> 31) == 0b0:
            vtype_str = self.decode_vtype(zimm)
            return f"vsetvli x{rd}, x{rs1}, {vtype_str}"
        elif (fields['raw'] >> 30) == 0b11:
            vtype_str = self.decode_vtype(zimm & 0x3FF)
            return f"vsetivli x{rd}, {rs1}, {vtype_str}"
        elif (fields['raw'] >> 25) == 0b1000000:
            return f"vsetvl x{rd}, x{rs1}, x{rs2}"

    def decode_vector_load(self, fields):
        """Decode vector load instructions"""
        width_str = self.width_map.get(fields['width'], f"e{fields['width']}")
        mop       = fields['mop']
        vm_str    = "" if fields['vm'] else ", v0.t"
        nf        = fields['nf']
        nf_str    = f"{nf+1}" if nf > 0 else ""
        
        self.debug_print(f"[DEBUG] mop: {mop:02b}")
        self.debug_print( f"[DEBUG] nf: {nf:01b}")
        self.debug_print(f"[DEBUG] width: {fields['width']:03b}")
        
        if mop == 0b00 and fields['lumop'] == 0b01000:
            return f"vl{nf}r{width_str}.v v{fields['vd']}, (x{fields['rs1']})"
                
        if mop == 0b00 and fields['lumop'] == 0b10000:
            if nf > 0:
                return f"vlseg{nf_str}e{width_str}ff.v v{fields['vd']}, (x{fields['rs1']}){vm_str}"
            else:
                return f"vle{width_str}ff.v v{fields['vd']}, (x{fields['rs1']}){vm_str}"
        
        if mop == 0b00 and fields['lumop'] == 0b01011:
            return f"vlm.v v{fields['vd']}, (x{fields['rs1']})"
        
        # Main load decoding
        if mop == 0b00 and fields['lumop'] == 0b00000:  
            if nf > 0:
                return f"vlseg{nf_str}e{width_str}.v v{fields['vd']}, (x{fields['rs1']}){vm_str}"
            else:
                return f"vle{width_str}.v v{fields['vd']}, (x{fields['rs1']}){vm_str}"
        elif mop == 0b10:  
            if nf > 0:
                return f"vlsseg{nf_str}e{width_str}.v v{fields['vd']}, (x{fields['rs1']}), x{fields['rs2']}{vm_str}"
            else:
                return f"vlse{width_str}.v v{fields['vd']}, (x{fields['rs1']}), x{fields['rs2']}{vm_str}"
        elif mop == 0b01: 
            if nf > 0:
                return f"vluxseg{nf_str}ei{width_str}.v v{fields['vd']}, (x{fields['rs1']}), v{fields['vs2']}{vm_str}"
            else:
                return f"vluxei{width_str}.v v{fields['vd']}, (x{fields['rs1']}), v{fields['vs2']}{vm_str}"
        elif mop == 0b11:  
            if nf > 0:
                return f"vloxseg{nf_str}ei{width_str}.v v{fields['vd']}, (x{fields['rs1']}), v{fields['vs2']}{vm_str}"
            else:
                return f"vloxei{width_str}.v v{fields['vd']}, (x{fields['rs1']}), v{fields['vs2']}{vm_str}"
        
        return f"Unknown vector load instruction"
        
    def decode_vector_store(self, fields):
        """Decode vector store instructions"""
        width_str = self.width_map.get(fields['width'], f"e{fields['width']}")
        mop       = fields['mop']
        vm_str    = "" if fields['vm'] else ", v0.t"
        nf        = fields['nf']
        nf_str    = f"{nf+1}" if nf > 0 else ""
        
        self.debug_print(f"[DEBUG] mop: {mop:02b}")
        self.debug_print( f"[DEBUG] nf: {nf:01b}")
        self.debug_print(f"[DEBUG] width: {fields['width']:03b}")
        
        if mop == 0b00 and fields['sumop'] == 0b01000:
            return f"vs{nf}r.v v{fields['vd']}, (x{fields['rs1']})"
       
        if mop == 0b00 and fields['sumop'] == 0b01011:
            return f"vsm.v v{fields['vd']}, (x{fields['rs1']})"
        
        # Main store decoding
        if mop == 0b00 and fields['sumop'] == 0b00000: 
            if nf > 0:
                return f"vsseg{nf_str}e{width_str}.v v{fields['vd']}, (x{fields['rs1']}){vm_str}"
            else:
                return f"vse{width_str}.v v{fields['vd']}, (x{fields['rs1']}){vm_str}"
        elif mop == 0b10:
            if nf > 0:
                return f"vssseg{nf_str}e{width_str}.v v{fields['vd']}, (x{fields['rs1']}), x{fields['rs2']}{vm_str}"
            else:
                return f"vsse{width_str}.v v{fields['vd']}, (x{fields['rs1']}), x{fields['rs2']}{vm_str}"
        elif mop == 0b01:
            if nf > 0:
                return f"vsuxseg{nf_str}ei{width_str}.v v{fields['vd']}, (x{fields['rs1']}), v{fields['vs2']}{vm_str}"
            else:
                return f"vsuxei{width_str}.v v{fields['vd']}, (x{fields['rs1']}), v{fields['vs2']}{vm_str}"
        elif mop == 0b11:
            if nf > 0:
                return f"vsoxseg{nf_str}ei{width_str}.v v{fields['vd']}, (x{fields['rs1']}), v{fields['vs2']}{vm_str}"
            else:
                return f"vsoxei{width_str}.v v{fields['vd']}, (x{fields['rs1']}), v{fields['vs2']}{vm_str}"
        
        return f"Unknown vector store instruction"

    def decode_vector_arithmetic(self, fields):
        """Decode vector arithmetic instructions using logical funct3-based approach"""
        config_result = self.decode_config_instruction(fields)
        if config_result:
            return config_result
            
        funct3 = fields['funct3']
        funct6 = fields['funct6']
        vm_str = "" if fields['vm'] else ", v0.t"
        
        self.debug_print(f"Arithmetic - funct3: {funct3:03b}, funct6: {funct6:06b}")
        
        # Handle special cases first
        special_result = self._decode_special_cases(fields, funct3, funct6, vm_str)
        if special_result:
            return special_result
        
        # Main instruction decoding based on funct3 category
        if funct3 in [self.OPIVV, self.OPIVX, self.OPIVI]:
            return self._decode_opi_instruction(fields, funct3, funct6, vm_str)
        elif funct3 in [self.OPMVV, self.OPMVX]:
            return self._decode_opm_instruction(fields, funct3, funct6, vm_str)
        elif funct3 in [self.OPFVV, self.OPFVF]:
            return self._decode_opf_instruction(fields, funct3, funct6, vm_str)
        
        return f"Unknown vector instruction (funct3={funct3:03b}, funct6={funct6:06b})"

    def _decode_special_cases(self, fields, funct3, funct6, vm_str):
        """Handle special case instructions"""
        if funct6 == 0b010000:  
            return self._decode_special_unary_0(fields, funct3, vm_str)
        elif funct6 == 0b010010: 
            return self._decode_special_unary_1(fields, funct3, vm_str)
        elif funct6 == 0b010011:  
            return self._decode_special_unary_2(fields, funct3, vm_str)
        elif funct6 == 0b010100:  
            return self._decode_mask_unary(fields, funct3, vm_str)
        elif funct6 == 0b100111 and funct3 == self.OPIVI:  
            return self._decode_vmv_nr_r(fields)
        
        return None

    def _decode_special_unary_0(self, fields, funct3, vm_str):
        """Decode special unary operations group 0"""
        if funct3 == self.OPMVX:
            if fields['vs2'] in self.vrxunary0_map:
                return f"{self.vrxunary0_map[fields['vs2']]} v{fields['vd']}, x{fields['rs1']}"
        elif funct3 == self.OPMVV:
            if fields['rs1'] in self.vwxunary0_map:
                return f"{self.vwxunary0_map[fields['rs1']]} x{fields['rd']}, v{fields['vs2']}{vm_str}"
        elif funct3 == self.OPFVF:
            if fields['vs2'] in self.vrfunary0_map:
                return f"{self.vrfunary0_map[fields['vs2']]} v{fields['vd']}, f{fields['rs1']}"
        elif funct3 == self.OPFVV:
            if fields['rs1'] in self.vwfunary0_map:
                return f"{self.vwfunary0_map[fields['rs1']]} f{fields['rd']}, v{fields['vs2']}"
        return None

    def _decode_special_unary_1(self, fields, funct3, vm_str):
        """Decode special unary operations group 1"""
        if funct3 == self.OPMVV:
            if fields['rs1'] in self.vxunary0_map:
                return f"{self.vxunary0_map[fields['rs1']]} v{fields['vd']}, v{fields['vs2']}{vm_str}"
        elif funct3 == self.OPFVV:
            if fields['rs1'] in self.vfunary0_map:
                return f"{self.vfunary0_map[fields['rs1']]} v{fields['vd']}, v{fields['vs2']}{vm_str}"
        return None

    def _decode_special_unary_2(self, fields, funct3, vm_str):
        """Decode special unary operations group 2"""
        if funct3 == self.OPFVV:
            if fields['rs1'] in self.vfunary1_map:
                return f"{self.vfunary1_map[fields['rs1']]} v{fields['vd']}, v{fields['vs2']}{vm_str}"
        return None

    def _decode_mask_unary(self, fields, funct3, vm_str):
        """Decode mask unary operations"""
        if funct3 == self.OPMVV:
            if fields['rs1'] in self.vmunary0_map:
                mnemonic = self.vmunary0_map[fields['rs1']]
                suffix = ".m" if fields['rs1'] in [0b00001, 0b00010, 0b00011, 0b10000] else ".v"
                vs2 = f", v{fields['vs2']}" if fields['rs1'] not in [0b10001] else ""
                return f"{mnemonic}{suffix} v{fields['vd']}{vs2}{vm_str}"
        return None

    def _decode_vmv_nr_r(self, fields):
        """Decode vmv<nr>r instruction - whole register moves"""
        simm_low = fields['imm'] & 0x7
        nreg = self.nfield_map.get(simm_low)
        
        if nreg is not None:
            return f"vmv{nreg}r.v v{fields['vd']}, v{fields['vs2']}"
        else:
            return f"vmv<nr>r v{fields['vd']}, v{fields['vs2']}  # Invalid simm[2:0]: {simm_low:03b}"
    
    def _decode_vmerge_vmv(self, fields, funct3, vm_str):
        """Decode vmerge/vmv instructions - they share funct6 but differ by vm bit"""
        if fields['vm'] == 0:
            if funct3 == self.OPIVV:
                return f"vmerge.vvm v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}, v0"
            elif funct3 == self.OPIVX:
                return f"vmerge.vxm v{fields['vd']}, v{fields['vs2']}, x{fields['rs1']}, v0"
            else: 
                imm = fields['imm']
                if imm & 0x10:
                    imm = imm - 0x20
                return f"vmerge.vim v{fields['vd']}, v{fields['vs2']}, {imm}, v0"
        else:
            if funct3 == self.OPIVV:
                return f"vmv.v.v v{fields['vd']}, v{fields['vs1']}"
            elif funct3 == self.OPIVX:
                return f"vmv.v.x v{fields['vd']}, x{fields['rs1']}"
            else:  
                imm = fields['imm']
                if imm & 0x10:
                    imm = imm - 0x20
                return f"vmv.v.i v{fields['vd']}, {imm}"
            
    def _decode_vfmerge_vfmv(self, fields, funct3, vm_str):
        """Decode vfmerge/vfmv instructions - they share funct6 but differ by vm bit"""
        if funct3 == self.OPFVF:
            if fields['vm'] == 0:
                return f"vfmerge.vfm v{fields['vd']}, v{fields['vs2']}, f{fields['rs1']}, v0"
            else:
                return f"vfmv.v.f v{fields['vd']}, f{fields['rs1']}"

    def _decode_opi_instruction(self, fields, funct3, funct6, vm_str):
        """Decode OPI instructions (Integer)"""
        if funct6 == 0b010111:  
            return self._decode_vmerge_vmv(fields, funct3, vm_str)
        
        mnemonic = self.opi_instructions.get(funct6)
        if not mnemonic:
            return None
            
        if funct3 == self.OPIVV:
            if mnemonic.startswith("vred") or mnemonic.startswith("vwred"):
                suffix = ".vs"
            elif mnemonic.startswith("vnclip"):
                suffix = ".wv"
            else: suffix = ".vv"
            operands = f"v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}"
        elif funct3 == self.OPIVX:
            if mnemonic.startswith("vnclip"):
                suffix = ".wx"
            else: suffix = ".vx"  
            operands = f"v{fields['vd']}, v{fields['vs2']}, x{fields['rs1']}"
        elif funct3 == self.OPIVI:
            if mnemonic.startswith("vnclip"):
                suffix = ".wi"
            else: suffix = ".vi"
            imm = fields['imm']
            if imm & 0x10:  
                imm = imm - 0x20
            operands = f"v{fields['vd']}, v{fields['vs2']}, {imm}"
        
        return f"{mnemonic}{suffix} {operands}{vm_str}"

    def _decode_opm_instruction(self, fields, funct3, funct6, vm_str):
        """Decode OPM instructions (Mask/Multiply)"""
        mnemonic = self.opm_instructions.get(funct6)
        if not mnemonic:
            return None
            
        if mnemonic == "vcompress":
            return f"vcompress.vm v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}"
        
        if funct3 == self.OPMVV:
            if mnemonic in ["vmacc", "vnmsac", "vmadd", "vnmsub", "vwmaccu", "vwmacc", "vwmaccsu"]:
                suffix = ".vv"
                operands = f"v{fields['vd']}, v{fields['vs1']}, v{fields['vs2']}"
            elif mnemonic in ["vmandn", "vmand", "vmor", "vmxor", "vmorn", "vmnand", "vmnor", "vmxnor"]:
                suffix = ".mm"
                operands = f"v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}"
            elif mnemonic.startswith("vred") or mnemonic.startswith("vwred"):
                suffix = ".vs"
                operands = f"v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}"
            elif mnemonic.startswith("vw") and mnemonic.endswith(".w"):
                mnemonic = mnemonic[:-2]
                suffix = ".wv"
                operands = f"v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}"
            else:
                suffix = ".vv"
                operands = f"v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}"
        else: 
            if mnemonic in ["vmacc", "vnmsac", "vmadd", "vnmsub", "vwmaccu", "vwmacc", "vwmaccsu", "vwmaccus"]:
                suffix = ".vx"
                operands = f"v{fields['vd']}, x{fields['rs1']}, v{fields['vs2']}"
            elif mnemonic.startswith("vw") and mnemonic.endswith(".w"):
                mnemonic = mnemonic[:-2]
                suffix = ".wx"
                operands = f"v{fields['vd']}, v{fields['vs2']}, x{fields['rs1']}"
            else:
                suffix = ".vx"
                operands = f"v{fields['vd']}, v{fields['vs2']}, x{fields['rs1']}"
        
        return f"{mnemonic}{suffix} {operands}{vm_str}"

    def _decode_opf_instruction(self, fields, funct3, funct6, vm_str):
        """Decode OPF instructions (Floating-Point)"""
        if funct6 == 0b010111:
            return self._decode_vfmerge_vfmv(fields, funct3, vm_str)
            
        mnemonic = self.opf_instructions.get(funct6)
        if not mnemonic:
            return None
            
        if funct3 == self.OPFVV:
            if mnemonic.startswith("vfw") and mnemonic.endswith(".w"):
                mnemonic = mnemonic[:-2]
                suffix = ".wv"
                operands = f"v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}"
            elif mnemonic.startswith("vfred") or mnemonic.startswith("vfwred"):
                suffix = ".vs"
                operands = f"v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}"
            else:
                suffix = ".vv"
                operands = f"v{fields['vd']}, v{fields['vs2']}, v{fields['vs1']}"
        else: 
            if mnemonic.startswith("vfw") and mnemonic.endswith(".w"):
                mnemonic = mnemonic[:-2]
                suffix = ".wf"
                operands = f"v{fields['vd']}, v{fields['vs2']}, f{fields['rs1']}"
            else:
                suffix = ".vf"
                operands = f"v{fields['vd']}, v{fields['vs2']}, f{fields['rs1']}"
        
        return f"{mnemonic}{suffix} {operands}{vm_str}"

    def decode_instruction(self, hex_str, debug=None):
        """Main decoding function"""
        original_debug = self.debug
        if debug is not None:
            self.debug = debug
            
        try:
            if hex_str.startswith('0x'):
                instruction = int(hex_str, 16)
            else:
                instruction = int(hex_str, 16)
            
            self.debug_print("=== STARTING DECODING ===")
            fields = self.extract_fields(instruction)
            if fields['opcode'] == self.LOAD_FP:
                result = self.decode_vector_load(fields)
            elif fields['opcode'] == self.STORE_FP:
                result = self.decode_vector_store(fields)
            elif fields['opcode'] == self.OP_V:
                result = self.decode_vector_arithmetic(fields)
                if not result:
                    result = f"Unknown vector instruction (funct3={fields['funct3']:03b}, funct6={fields['funct6']:06b})"
            else:
                result = f"Non-vector instruction (opcode=0x{fields['opcode']:02x})"
                
            self.debug_print(f"=== FINAL RESULT: {result} ===")
            return result
                
        except ValueError:
            error_msg = f"Invalid hex input: {hex_str}"
            self.debug_print(f"=== ERROR: {error_msg} ===")
            return error_msg
        except Exception as e:
            error_msg = f"Error decoding instruction: {e}"
            self.debug_print(f"=== ERROR: {error_msg} ===")
            return error_msg
        finally:
            self.debug = original_debug

def process_file(input_file, output_file=None, debug=False):
    """Process a file, replacing all 8-hex-digit strings with decoded instructions"""
    decoder = RVVDecoder(debug=debug)

    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return 1
    except Exception as e:
        print(f"Error reading input file: {e}")
        return 1

    # More precise pattern to avoid false positives
    hex_pattern = r'\b[0-9a-fA-F]{8}\b'
    
    def replace_hex(match):
        hex_str = match.group(0)
        try:
            # Quick validation - ensure it's a valid 32-bit integer
            instruction_val = int(hex_str, 16)
            if instruction_val > 0xFFFFFFFF:
                return hex_str  # Not a valid instruction, keep original
                
            decoded = decoder.decode_instruction(hex_str)
            
            # If it's not a vector instruction, keep the original
            if "Non-vector instruction" in decoded or "Unknown" in decoded:
                return hex_str
                
            return f"{decoded}"
        except (ValueError, Exception):
            # If decoding fails, return the original hex
            return hex_str

    processed_content = re.sub(hex_pattern, replace_hex, content)
    
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(processed_content)
            print(f"Successfully processed {input_file} -> {output_file}")
            return 0
        except Exception as e:
            print(f"Error writing output file: {e}")
            return 1
    else:
        print(processed_content)
        return 0

def main():
    parser = argparse.ArgumentParser(description='RISC-V Vector Instruction Decoder')
    parser.add_argument('-i', '--input', help='Input file to process')
    parser.add_argument('-o', '--output', help='Output file (optional)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
    parser.add_argument('hex_string', nargs='?', help='Single hex string to decode (8 digits, no 0x prefix)')
    
    args = parser.parse_args()
    
    if args.input:
        return process_file(args.input, args.output, args.debug)
    elif args.hex_string:
        decoder = RVVDecoder(debug=args.debug)
        result = decoder.decode_instruction(args.hex_string)
        print(f"Decoded: {result}")
        return 0
    else:
        decoder = RVVDecoder(debug=args.debug)
        
        print("RISC-V Vector Instruction Decoder")
        print("Enter hex machine code (8 digits, no 0x prefix) or 'quit' to exit")
        
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_input:
                    continue
                    
                result = decoder.decode_instruction(user_input)
                print(f"Decoded: {result}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                break

if __name__ == "__main__":
    main()
