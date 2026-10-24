# Writes a throwaway certificate authority, a server certificate for 127.0.0.1 and a client
# certificate into a directory, as PEM files, for the mutual TLS tests.

import datetime
import ipaddress
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


@dataclass(frozen=True)
class TlsFiles:
    ca: Path
    server_cert: Path
    server_key: Path
    client_cert: Path
    client_key: Path


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _write_key(key: ec.EllipticCurvePrivateKey, path: Path) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _issue(
    subject: str,
    key: ec.EllipticCurvePrivateKey,
    issuer: x509.Name,
    signer: ec.EllipticCurvePrivateKey,
    authority: bool,
    address: str | None,
) -> x509.Certificate:
    now = datetime.datetime.now(datetime.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(subject))
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=authority, path_length=None), critical=True)
    )
    if address is not None:
        names = [x509.IPAddress(ipaddress.ip_address(address))]
        builder = builder.add_extension(x509.SubjectAlternativeName(names), critical=False)
    return builder.sign(signer, hashes.SHA256())


def write_tls_files(directory: Path) -> TlsFiles:
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = _issue("test ca", ca_key, _name("test ca"), ca_key, True, None)
    server_key = ec.generate_private_key(ec.SECP256R1())
    server = _issue("server", server_key, ca.subject, ca_key, False, "127.0.0.1")
    client_key = ec.generate_private_key(ec.SECP256R1())
    client = _issue("client", client_key, ca.subject, ca_key, False, None)
    files = TlsFiles(
        directory / "ca.pem",
        directory / "server.pem",
        directory / "server-key.pem",
        directory / "client.pem",
        directory / "client-key.pem",
    )
    for certificate, path in (
        (ca, files.ca),
        (server, files.server_cert),
        (client, files.client_cert),
    ):
        path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    _write_key(server_key, files.server_key)
    _write_key(client_key, files.client_key)
    return files
