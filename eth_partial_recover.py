#!/usr/bin/env python3
# eth_partial_recover.py
# -----------------------------------------------------------------------------
# Referencia/oraculo en Python para la recuperacion parcial de clave ETH.
# Usa librerias verificadas (correccion garantizada) para:
#   1) confirmar los vectores del self-test del programa CUDA,
#   2) validar tu sufijo + direccion (probando un prefijo conocido),
#   3) hacer busquedas PEQUENAS de validacion (lento, no para el espacio real).
#
# Instalacion:
#   pip install coincurve pycryptodome
# -----------------------------------------------------------------------------
import sys

try:
    from coincurve import PublicKey
    from Crypto.Hash import keccak
except ImportError:
    print("Faltan dependencias. Ejecuta:  pip install coincurve pycryptodome")
    sys.exit(1)

def keccak256(b: bytes) -> bytes:
    h = keccak.new(digest_bits=256)
    h.update(b)
    return h.digest()

def priv_to_address(priv: bytes) -> str:
    """priv: 32 bytes -> direccion ETH '0x...' (minusculas)."""
    pub = PublicKey.from_valid_secret(priv).format(compressed=False)  # 65 bytes, 0x04||x||y
    addr = keccak256(pub[1:])[-20:]
    return "0x" + addr.hex()

def full_key_from_parts(prefix_hex12: str, suffix_hex52: str) -> bytes:
    h = prefix_hex12 + suffix_hex52
    assert len(h) == 64, "la clave completa debe tener 64 hex"
    return bytes.fromhex(h)

def selftest_vectors():
    print("== Vectores de validacion (deben coincidir con el self-test CUDA) ==")
    print("keccak256(\"\") =", keccak256(b"").hex())
    print("  esperado    = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
    p1 = (1).to_bytes(32, "big"); p2 = (2).to_bytes(32, "big")
    print("addr(priv=1)  =", priv_to_address(p1))
    print("  esperado    = 0x7e5f4552091a69125d5dfcb7b8c2659029395bdf")
    print("addr(priv=2)  =", priv_to_address(p2))
    print("  esperado    = 0x2b5ad5c4795c026514f8317c7a215e218dccd6cf")

def validate_inputs(suffix_hex52: str, target_addr: str):
    """Confirma que tu sufijo/direccion estan bien formados probando prefijo=0."""
    suffix_hex52 = suffix_hex52.lower()
    target_addr = target_addr.lower()
    if target_addr.startswith("0x"): target_addr = target_addr[2:]
    assert len(suffix_hex52) == 52, "el sufijo debe ser 52 hex"
    assert len(target_addr) == 40, "la direccion debe ser 40 hex"
    test_prefix = "000000000000"
    addr = priv_to_address(full_key_from_parts(test_prefix, suffix_hex52))
    print(f"Formato OK. Con prefijo {test_prefix} la direccion seria {addr}")
    print(f"Objetivo: 0x{target_addr}")
    if addr[2:] == target_addr:
        print(">>> El prefijo es justo 000000000000 (caso afortunado).")

def small_search(suffix_hex52: str, target_addr: str, max_prefixes: int):
    """Busqueda lineal pequena en Python (validacion). NO para 2^48."""
    target_addr = target_addr.lower()
    if target_addr.startswith("0x"): target_addr = target_addr[2:]
    suf = int(suffix_hex52, 16)
    for p in range(max_prefixes):
        priv = ((p << 208) | suf).to_bytes(32, "big")
        if priv_to_address(priv)[2:] == target_addr:
            print(f"ENCONTRADO prefijo = {p:012x}")
            print(f"clave privada = {p:012x}{suffix_hex52}")
            return
        if p % 5000 == 0 and p:
            print(f"  ...{p} prefijos probados", end="\r")
    print("\nno encontrado en ese rango pequeno (es solo validacion).")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest_vectors()
    elif sys.argv[1] == "validate" and len(sys.argv) == 4:
        validate_inputs(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "search" and len(sys.argv) == 5:
        small_search(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        print("Uso:")
        print("  python3 eth_partial_recover.py                      # imprime vectores")
        print("  python3 eth_partial_recover.py validate <sufijo52> <addr>")
        print("  python3 eth_partial_recover.py search   <sufijo52> <addr> <max_prefijos>")
