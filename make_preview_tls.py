"""Create a short-lived certificate for the private LAN preview."""

import argparse
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def write_certificates(address, directory):
    public = directory / "public"
    public.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "6301 Preview Test CA")])
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, str(address))])

    def certificate(subject, issuer, key, days):
        return (x509.CertificateBuilder()
                .subject_name(subject).issuer_name(issuer).public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=5))
                .not_valid_after(now + timedelta(days=days)))

    ca = (certificate(ca_name, ca_name, ca_key, 7)
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, None, None), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    leaf = (certificate(server_name, ca_name, server_key, 6)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(address)]), critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.KeyUsage(True, False, True, False, False, False, False, None, None), critical=True)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))

    (public / "preview-ca.cer").write_bytes(ca.public_bytes(serialization.Encoding.DER))
    (directory / "preview-ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    (directory / "server.pem").write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key = directory / "server-key.pem"
    key.write_bytes(server_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                            serialization.NoEncryption()))
    key.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", type=ip_address)
    parser.add_argument("--directory", type=Path, default=Path("runs/preview-tls"))
    args = parser.parse_args()
    if (args.directory / "public" / "preview-ca.cer").exists():
        parser.error("Use a new directory to avoid replacing a CA already installed on a device")
    write_certificates(args.address, args.directory)


if __name__ == "__main__":
    main()
