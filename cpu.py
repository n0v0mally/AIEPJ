import sys
import json
import struct

# --- 常量定义 ---
AOK, HLT, ADR, INS = 1, 2, 3, 4

RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI = 0, 1, 2, 3, 4, 5, 6, 7
R8, R9, R10, R11, R12, R13, R14, RNONE = 8, 9, 10, 11, 12, 13, 14, 15

I_HALT, I_NOP, I_RRMOVQ, I_IRMOVQ = 0, 1, 2, 3
I_RMMOVQ, I_MRMOVQ, I_ALU, I_JMP = 4, 5, 6, 7
I_CALL, I_RET, I_PUSHQ, I_POPQ = 8, 9, 10, 11


class Y86Simulator:
    def __init__(self, mem_size=8192):
        self.REG_NAMES = {
            0: "rax", 1: "rcx", 2: "rdx", 3: "rbx", 4: "rsp", 5: "rbp", 6: "rsi", 7: "rdi",
            8: "r8", 9: "r9", 10: "r10", 11: "r11", 12: "r12", 13: "r13", 14: "r14"
        }
        self.mem_size = mem_size
        self.memory = bytearray(mem_size)
        self.PC = 0
        self.STAT = AOK

        self.REG = {i: 0 for i in range(15)}
        self.REG[RNONE] = 0

        self.CC = {"ZF": 1, "SF": 0, "OF": 0}

        # CPU 内部暂存器 / 数据通路状态变量 (用于在多周期状态机的各阶段间传递当前指令的数据)
        self.icode = I_NOP
        self.ifun = 0
        self.rA = RNONE
        self.rB = RNONE
        self.valC = 0
        self.valP = 0
        self.valA = 0
        self.valB = 0
        self.valE = 0
        self.valM = 0
        self.Cnd = False

        self.trace = []

    # ---------------- 辅助方法 ----------------
    def to_signed64(self, val):
        """将无符号 64 位整数截断并转换为有符号补码整数"""
        val = val & 0xFFFFFFFFFFFFFFFF
        if val & 0x8000000000000000:
            val -= 0x10000000000000000
        return val

    def sign_extend(self, val, bits=64):
        """将指定位数的无符号整数转换为补码有符号整数"""
        sign_bit = 1 << (bits - 1)
        return (val & (sign_bit - 1)) - (val & sign_bit)

    def uint64(self, val):
        """将数值截断为 64 位无符号整数"""
        return val & 0xFFFFFFFFFFFFFFFF

    def read_mem(self, addr, size):
        """从指定内存地址读取指定字节数的数据，若地址越界则触发 ADR 异常"""
        if addr < 0 or addr + size > self.mem_size:
            self.STAT = ADR
            return 0
        data = self.memory[addr: addr + size]
        if size == 1:
            return data[0]
        elif size == 8:
            return struct.unpack('<Q', data)[0]
        return 0

    def write_mem(self, addr, size, val):
        """将数据写入指定的内存地址，若地址越界则触发 ADR 异常"""
        if addr < 0 or addr + size > self.mem_size:
            self.STAT = ADR
            return
        if size == 8:
            self.memory[addr: addr + size] = struct.pack('<Q', self.uint64(val))

    def check_cond(self, ifun):
        """根据当前条件码寄存器，判断跳转或条件传送是否满足"""
        zf, sf, of = self.CC["ZF"], self.CC["SF"], self.CC["OF"]
        if ifun == 0: return True
        if ifun == 1: return (sf ^ of) | zf
        if ifun == 2: return sf ^ of
        if ifun == 3: return zf
        if ifun == 4: return not zf
        if ifun == 5: return not (sf ^ of)
        if ifun == 6: return not (sf ^ of) and not zf
        return False

    # ---------------- 核心 FSM 逻辑 ----------------

    def fetch(self):
        """
        取指阶段 (Fetch)

        读入 1 字节的 icode/ifun，再按指令格式决定后面是否跟一个寄存器字节、
        是否跟 8 字节立即数，最后算出下一条指令的地址 valP。
        """
        if self.PC < 0 or self.PC >= self.mem_size:
            self.STAT = ADR
            return

        byte0 = self.read_mem(self.PC, 1)
        if self.STAT != AOK:
            return

        self.icode = byte0 >> 4
        self.ifun = byte0 & 0x0F

        # icode 只有 0x0 ~ 0xB 是合法的（0xF 之类的硬编码字节要报 INS）
        if self.icode > I_POPQ:
            self.STAT = INS
            return

        # halt 是一条“没有后续阶段”的指令，取指阶段直接停机
        if self.icode == I_HALT:
            self.STAT = HLT
            return

        # 哪些指令带寄存器指示符字节 / 8 字节常数
        # jXX 与 call 只有 Dest(8 字节) 段，没有寄存器指示符字节
        need_reg = self.icode in (I_RRMOVQ, I_IRMOVQ, I_RMMOVQ, I_MRMOVQ,
                                  I_ALU, I_PUSHQ, I_POPQ)
        need_valC = self.icode in (I_IRMOVQ, I_RMMOVQ, I_MRMOVQ, I_JMP, I_CALL)

        self.rA = RNONE
        self.rB = RNONE
        self.valC = 0

        addr = self.PC + 1
        if need_reg:
            b = self.read_mem(addr, 1)
            if self.STAT != AOK:
                return
            self.rA = b >> 4
            self.rB = b & 0x0F
            addr += 1

        if need_valC:
            self.valC = self.read_mem(addr, 8)
            if self.STAT != AOK:
                return
            addr += 8

        self.valP = addr

    def decode(self):
        """
        译码阶段 (Decode)

        根据指令选择两个源寄存器 srcA / srcB，并从寄存器文件读出 valA / valB。
        栈类指令会用隐式的 %rsp 作为操作数。
        """
        srcA = RNONE
        srcB = RNONE

        if self.icode == I_RRMOVQ:          # rrmovq / cmovXX
            srcA = self.rA
        elif self.icode == I_RMMOVQ:       # rmmovq rA, D(rB)
            srcA = self.rA
            srcB = self.rB
        elif self.icode == I_MRMOVQ:       # mrmovq D(rB), rA
            srcB = self.rB
        elif self.icode == I_ALU:          # OPq rA, rB
            srcA = self.rA
            srcB = self.rB
        elif self.icode in (I_CALL, I_RET, I_PUSHQ, I_POPQ):
            # 这四条都要读写栈指针，%rsp 是隐式的第二操作数
            srcB = RSP
            if self.icode == I_PUSHQ:
                srcA = self.rA
            elif self.icode in (I_RET, I_POPQ):
                # popq / ret 读内存用的是“弹出的旧栈顶”，也就是 valA
                srcA = RSP

        self.valA = self.REG[srcA] if srcA != RNONE else 0
        self.valB = self.REG[srcB] if srcB != RNONE else 0

    def execute(self):
        """
        执行阶段 (Execute)

        ALU 算出有效地址或运算结果 valE；OPq 更新条件码；跳转/条件传送算出 Cnd。
        """
        self.valE = 0
        self.Cnd = False

        aluA = 0
        aluB = 0

        if self.icode == I_RRMOVQ:
            aluA, aluB = 0, self.valA
            self.Cnd = self.check_cond(self.ifun)
        elif self.icode == I_IRMOVQ:
            aluA, aluB = self.valC, 0
        elif self.icode in (I_RMMOVQ, I_MRMOVQ):
            aluA, aluB = self.valC, self.valB
        elif self.icode == I_ALU:
            aluA, aluB = self.valA, self.valB
        elif self.icode == I_JMP:
            self.Cnd = self.check_cond(self.ifun)
        elif self.icode in (I_CALL, I_PUSHQ):
            aluA, aluB = -8, self.valB
        elif self.icode in (I_RET, I_POPQ):
            aluA, aluB = 8, self.valB

        if self.icode == I_ALU:
            a = self.sign_extend(self.uint64(aluA))
            b = self.sign_extend(self.uint64(aluB))
            if self.ifun == 0:          # addq
                full = a + b
                self.valE = self.uint64(full)
                res = self.sign_extend(self.valE)
                self.CC["ZF"] = 1 if res == 0 else 0
                self.CC["SF"] = 1 if res < 0 else 0
                self.CC["OF"] = 1 if full > 0x7FFFFFFFFFFFFFFF or full < -0x8000000000000000 else 0
            elif self.ifun == 1:        # subq：注意 Y86 是 aluB - aluA
                full = b - a
                self.valE = self.uint64(full)
                res = self.sign_extend(self.valE)
                self.CC["ZF"] = 1 if res == 0 else 0
                self.CC["SF"] = 1 if res < 0 else 0
                self.CC["OF"] = 1 if full > 0x7FFFFFFFFFFFFFFF or full < -0x8000000000000000 else 0
            elif self.ifun == 2:        # andq
                self.valE = self.uint64(aluA & aluB)
                res = self.sign_extend(self.valE)
                self.CC["ZF"] = 1 if res == 0 else 0
                self.CC["SF"] = 1 if res < 0 else 0
                self.CC["OF"] = 0
            elif self.ifun == 3:        # xorq
                self.valE = self.uint64(aluA ^ aluB)
                res = self.sign_extend(self.valE)
                self.CC["ZF"] = 1 if res == 0 else 0
                self.CC["SF"] = 1 if res < 0 else 0
                self.CC["OF"] = 0
        else:
            # 地址计算 / 立即数装载 / 栈指针加减：一律做加法，且不影响条件码
            self.valE = self.uint64(aluB + aluA)

    def memory_stage(self):
        """
        访存阶段 (Memory)

        rmmovq / pushq / call 写内存；mrmovq / popq / ret 读内存。
        越界时 write_mem / read_mem 会把 STAT 置成 ADR。
        """
        self.valM = 0

        if self.icode == I_RMMOVQ:
            self.write_mem(self.valE, 8, self.valA)
        elif self.icode == I_PUSHQ:
            # pushq %rsp 压入的是 valA —— 也就是压栈前读到的旧值
            self.write_mem(self.valE, 8, self.valA)
        elif self.icode == I_CALL:
            self.write_mem(self.valE, 8, self.valP)
        elif self.icode == I_MRMOVQ:
            self.valM = self.read_mem(self.valE, 8)
        elif self.icode in (I_POPQ, I_RET):
            # 读的是压栈时的栈顶（valA），而 valE 是 +8 之后的新栈指针
            self.valM = self.read_mem(self.valA, 8)

    def write_back(self):
        """
        写回阶段 (Write Back)

        dstE 写 valE，dstM 写 valM。先写 dstE 再写 dstM，这样 popq %rsp 时
        从内存读出的值才能覆盖掉“栈指针 +8”的结果。
        """
        dstE = RNONE
        dstM = RNONE

        if self.icode == I_RRMOVQ:
            if self.Cnd:
                dstE = self.rB
        elif self.icode == I_IRMOVQ:
            dstE = self.rB
        elif self.icode == I_ALU:
            dstE = self.rB
        elif self.icode == I_MRMOVQ:
            dstM = self.rA
        elif self.icode in (I_CALL, I_RET, I_PUSHQ, I_POPQ):
            dstE = RSP
            if self.icode == I_POPQ:
                dstM = self.rA

        if dstE != RNONE:
            self.REG[dstE] = self.uint64(self.valE)
        if dstM != RNONE:
            self.REG[dstM] = self.uint64(self.valM)

    def update_pc(self):
        """
        更新 PC 阶段 (Update PC)
        """
        if self.icode == I_CALL:
            self.PC = self.valC
        elif self.icode == I_JMP:
            self.PC = self.valC if self.Cnd else self.valP
        elif self.icode == I_RET:
            self.PC = self.valM
        else:
            self.PC = self.valP

    # ---------------- JSON 序列化输出 ----------------
    def dump_state(self):
        """将当前状态转存为字典"""
        mem_dump = {}
        for addr in range(0, self.mem_size, 8):
            val = struct.unpack('<q', self.memory[addr:addr + 8])[0]
            if val != 0:
                mem_dump[str(addr)] = val

        state = {
            "CC": dict(self.CC),
            "MEM": mem_dump,
            "PC": self.PC,
            "REG": {self.REG_NAMES[k]: self.to_signed64(v) for k, v in sorted(self.REG.items()) if k != RNONE},
            "STAT": self.STAT
        }
        self.trace.append(state)

    # ---------------- 入口流转 ----------------
    def load_program(self, file_obj):
        """解析器 Parser"""
        for line in file_obj:
            idx = line.find('|')
            code_part = line[:idx].strip() if idx != -1 else line.strip()
            if not code_part: continue
            if ':' in code_part:
                addr_str, data_str = code_part.split(':', 1)
                addr_str, data_str = addr_str.strip(), data_str.strip()
                if addr_str and data_str:
                    addr = int(addr_str, 16)
                    data_bytes = bytes.fromhex(data_str)
                    for i, b in enumerate(data_bytes):
                        if addr + i < len(self.memory):
                            self.memory[addr + i] = b

    def run(self):
        """主循环：时钟步进"""
        while self.STAT == AOK:
            self.fetch()
            if self.STAT != AOK:
                self.dump_state()
                break

            self.decode()
            self.execute()
            self.memory_stage()

            self.write_back()

            if self.STAT != AOK:
                self.dump_state()
                break

            self.update_pc()
            self.dump_state()

        print(json.dumps(self.trace, indent=4))


if __name__ == "__main__":
    sim = Y86Simulator()
    sim.load_program(sys.stdin)
    sim.run()
