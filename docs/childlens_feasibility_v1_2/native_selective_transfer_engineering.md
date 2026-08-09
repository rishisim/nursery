# ChildLens v1.2 native selective-transfer engineering

Version: `childlens-native-transfer-engineering-v1.2.0`  
Date: 2026-07-21  
Status: `CONTROLLER_VALIDATED; NATIVE_AUTHENTICATION_NOT_EXERCISED; NO_DATA_ACQUIRED`

## Result

The repository now contains a fail-closed controller for transferring only the
15 objects already frozen by the v1.1 content-blind selection. It accepts the
existing restricted v1.1 plan only when its canonical digest matches a second
restricted configuration binding, performs either an exact native-metadata
admission or the predeclared conservative storage-only amendment, and streams
one object at a time into the existing local quarantine.

The controller was tested only with synthetic byte streams. This work did not
authenticate to Keeper, inspect its UI, access email, issue a data request, or
download ChildLens material. It did not alter v1/v1.1 artifacts, selection,
permissions, scientific gates, or thresholds.

## Official native routes

[Keeper 7.0 is officially described by MPDL](https://www.mpdl.mpg.de/en/about-us/news/1100-keeper-7-0-new-features-and-improved-user-experience.html)
as being based on Seafile Professional 12.0.16. The relevant read-only Seafile
Web API routes are:

- [`GET /api2/repos/{repo_id}/file/detail/?p=...`](https://seafile-api.readme.io/reference/get_api2-repos-repo-id-file-detail), which returns file detail; the official older reference shows the response's integer `size` field;
- [`GET /api2/repos/{repo_id}/file/?p=...`](https://seafile-api.readme.io/reference/get_api2-repos-repo-id-file), which returns a link for downloading the current file; and
- [`GET /api2/repos/{repo_id}/file/revision/?p=...&commit_id=...`](https://seafile-api.readme.io/reference/get_api2-repos-repo-id-file-revision), which returns a download link for a named revision.

The controller supports `FILE_DETAIL` for a live revision and
`DOWNLOAD_HEAD` for a response-header `Content-Length`. When an immutable
commit is supplied, it requires `DOWNLOAD_HEAD` and uses the revision endpoint;
it refuses to represent current-file detail as commit-bound metadata.

Seafile's official [security documentation](https://manual.seafile.com/latest/administration/security_features/)
states that version 12 fileserver URLs authorize using cookies or API tokens,
whereas older servers used random, normally one-use URLs. Consequently, a
browser-authenticated Keeper session is not automatically a credential for an
external local controller. Transferring the session to this controller would
require exporting a cookie, API token, or bearer download URL. This engineering
track did none of those things. A normal browser-click download would also lack
the controller's hard counter, quarantine-only atomic promotion, and receipt
boundary.

No official local Seafile client or `seaf-cli` executable was present in the
read-only host check. Seafile's official
[authentication reference](https://seafile-api.readme.io/reference/authentication)
says version 11 and later use `Bearer`, documents a library-scoped repo token
with read-only or read/write permission, and says such tokens can be generated
in the library's Web UI under **Advanced → API Token**. A read-only repo token,
supplied through a fixed local environment variable, is therefore the
least-privilege native path for this version. The token is never accepted on
the command line, persisted, printed, or included in an error. `Token` remains
supported only for older documented servers through an explicit restricted
configuration choice.

## Admission modes

### Native exact

`NATIVE_EXACT` first obtains all 15 sizes without opening media bodies. It
admits only a positive total no greater than 21,474,836,479 bytes. If an
immutable revision is configured, every metadata HEAD and later GET uses the
revision-download endpoint. Response `Content-Length`, when present, must equal
the preflight size; otherwise the stream must terminate at exactly the
preflight size.

### Conservative storage-only amendment

`CONSERVATIVE_ROUNDED` is allowed only when the restricted configuration
contains exactly one bound for each frozen object and attests that the
amendment was frozen before media opening. For object `i`, the controller uses:

```text
upper_i = rounded_display_bytes_i
        + worst_case_rounding_quantum_bytes_i
        + transfer_overhead_bytes_i
```

The coordinator-frozen aggregate is 14,970,186,240 bytes for all 15 objects.
It is below the conservative admission ceiling of 18 GiB
(19,327,352,832 bytes), retaining a 2 GiB margin beneath the 20 GiB raw cap.
This amendment changes storage admission only. It does not change selection,
scientific gates, or thresholds. The stream aborts before crossing 20 GiB; the
largest allowed cumulative value is exactly 21,474,836,479 bytes. It also
aborts if any individual response exceeds its predeclared conservative bound.

Both modes remeasure allocated quarantine bytes and volume free space. They
fail if the proposed raw bound plus the predeclared nonraw reserve would exceed
the 73 GiB namespace cap or leave less than 50 GiB free.

## Restricted execution boundary

The controller requires:

- an existing owner-only quarantine outside the repository with no symlink in
  the supplied root path;
- private `0600` restricted plan and configuration files inside that root;
- affirmative restricted attestations for owner-only access, Git separation,
  Spotlight exclusion, backup exclusion or approved encrypted-local-only
  equivalence, signed-agreement controls, and the July 31, 2027 retention
  deadline;
- exactly 15 sequentially ranked v1.1 plan rows with unresolved size/checksum
  fields and no mutation of the frozen plan digest; and
- HTTPS API and generated-download origins from a small explicit allowlist.

The generated download link is held only in memory. Redirects are restricted
to the same download-origin allowlist; credentials are never forwarded unless
the restricted configuration explicitly permits it. API JSON bodies are
bounded, parsed in memory, and never saved. Unknown file-detail fields—such as
names or actor metadata—are ignored.

The frozen v1.1 locators retain the release-level `/ChildLens` namespace
label, while Keeper's repo-scoped Seafile API is already rooted at that same
library. A deterministic transport mapping therefore removes only that fixed
leading namespace component before a read-only API request. This mapping was
validated with metadata only after authentication, before any media body was
opened. It does not alter the frozen object key, basename, rank, participant,
selection hash, or scientific inclusion rule.

For Keeper's documented Seafile 12 base, the restricted configuration should
use `Bearer`. Set `send_authorization_to_download` only when the generated
fileserver link is on the exact allowlisted Keeper origin and actually requires
the repo token. Older one-use links should leave that flag false, preventing
credential forwarding to a fileserver link that does not need it.

Each object is streamed into a same-filesystem `0600` temporary file. The
controller enforces the per-object and hard cumulative counters before writing
each chunk, fsyncs the completed file, calculates SHA-256, and atomically
promotes it under a content-addressed opaque name. Byte-identical selected
objects refer to one stored content object rather than duplicate full-resolution
copies. A partial file is removed on controlled failure. A completed restricted
receipt is verified before resume.

The restricted receipt retains per-object rank, opaque object key, exact byte
count, SHA-256, ETag, content-length observation, and content-addressed relative
path. The reportable receipt contains only the already reportable release and
frozen download-plan digests, total counts/bytes, broad all-or-none header
coverage, admission/cap constants, and explicit restricted-payload absence
flags. The more granular restricted selection digest remains quarantined.
Positive partial counts below five and their byte/content totals are suppressed.

## Credential-safe launcher

`scripts/launch_childlens_native_transfer_v1_2.py` is the preferred execution
surface. It takes no arguments. It scans only owner-private, ChildLens-labelled
hidden roots immediately below the repository parent, and within those roots
opens only bounded private JSON candidates whose filename or parent context is
shaped like a download/transfer plan or transfer configuration. This includes
the coordinator shape `transfer_v1_2/config.json` without opening unrelated
annotation JSON. It proceeds only when exactly one plan/config pair passes the
controller schemas, shares one quarantine root, matches by the frozen canonical
plan digest, uses Keeper's exact HTTPS hostname with `Bearer`, and reproduces
the frozen 14,970,186,240-byte conservative bound. Multiple matches fail
closed. The namespace root is the deepest common owner-only ancestor containing
both matched files, so capacity accounting covers their actual shared
quarantine rather than automatically widening to the top hidden directory.

The temporary credential is read from macOS Keychain using the fixed service
`ChildLens-v1.2-Keeper-Repo-Token` and account
`childlens-v1.2-read-only`. It is placed in the controller's fixed environment
variable only for the two in-process calls, then removed or restored. The
launcher emits only fixed states. It never accepts or prints an argument,
credential, path, locator, response, or restricted diagnostic.

The launcher deletes that exact Keychain item only after `prepare` returns
`PREPARED` and `acquire` returns a verified 15-of-15 `COMPLETE` aggregate. A
failed transfer or unexpected state retains the Keychain item for a controlled
retry. Failure to delete after an otherwise complete transfer is reported as a
failure, not silently treated as success.

Retries are receipt-aware. A valid restricted `PREPARED`, `IN_PROGRESS`, or
`COMPLETE` receipt resumes through `acquire` without overwriting progress with
a fresh preparation. An invalid existing receipt fails before Keychain access;
it is never silently replaced.

## Invocation contract

After creating a read-only Keeper repo token in **Advanced → API Token**, add it
to Keychain with a prompt rather than a command-line value:

```text
/usr/bin/security add-generic-password -U \
  -s ChildLens-v1.2-Keeper-Repo-Token \
  -a childlens-v1.2-read-only \
  -w
```

`security` documents that placing `-w` last prompts for the secret. Paste the
token only into that local prompt. Then run:

```text
python3 scripts/launch_childlens_native_transfer_v1_2.py
```

The launcher calls the lower-level controller's `prepare` and `acquire` phases
with the uniquely discovered restricted paths. The direct controller remains
available for controlled engineering diagnosis, but it should not receive a
token or restricted path on its command line.

Success prints only the phase and state. Failure prints one constant error
code. The command never prints filenames, locators, identifiers, paths,
download links, response bodies, ETags, hashes, or authorization material.

## Validation

Synthetic test command:

```text
pytest -q \
  tests/test_childlens_native_transfer_v1_2.py \
  tests/test_launch_childlens_native_transfer_v1_2.py
```

Result: `25 passed`.

The tests cover immutable 15-row plan binding, plan-mutation rejection, the
coordinator-frozen conservative total, the 18 GiB ceiling and exact hard abort
value, exact metadata preparation, one-stream-at-a-time acquisition,
content-addressed atomic promotion, duplicate-content storage collapse,
pre-body oversize abort, restricted receipt tamper detection, public small-cell
suppression, HTTPS/origin enforcement, official detail/revision route
construction, repository-boundary rejection, and public sentinel absence.
They also verify that malformed command-line input produces only a constant
error and cannot echo a restricted argument.

The launcher tests additionally cover unique private-bundle discovery,
cross-digest matching, ambiguous-bundle refusal, strict Keeper/Bearer/frozen
bound enforcement, token presence only during the two controller calls,
environment restoration, deletion only after a verified complete result,
retention on failure, exact fixed Keychain service/account commands, captured
secret output, and deletion-failure handling. No synthetic test made a network
request or used the real Keychain item.
Argument rejection is separately sentinel-tested to ensure an accidental path
or credential is never echoed and never starts discovery.
The exact `transfer_v1_2/config.json` coordinator shape and deepest-common-root
rule have a dedicated synthetic test.
Valid prepared-receipt resume and invalid-receipt pre-Keychain refusal are also
tested.

## Disposition

The transfer engineering itself is ready. Actual native acquisition still
requires one user-only authentication action: create a **read-only**,
library-scoped repo API token in Keeper's library menu if that control is
available to this account, and place it locally in the fixed environment
variable for the authorized run. Do not paste it into chat or a repository
file. If the shared-library role cannot create such a token, the user must say
only that the menu is unavailable; no account password, cookie, or token should
be disclosed. Web login alone does not provide the external-controller
capability under the documented Seafile 12 design.
