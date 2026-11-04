const FEEDBACK_TO = "othmanqderat200@gmail.com";
const MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024;
const MAX_REQUEST_BYTES = MAX_ATTACHMENT_BYTES + 32 * 1024;
const ALLOWED_CATEGORIES = new Set([
  "Bug report",
  "Improvement suggestion",
  "Question",
  "Other",
]);
const MIME_TYPES: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
};

function jsonResponse(status: number, body: Record<string, unknown>) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function extensionOf(filename: string): string {
  const index = filename.lastIndexOf(".");
  return index < 0 ? "" : filename.slice(index).toLowerCase();
}

function hasValidImageSignature(extension: string, bytes: Uint8Array): boolean {
  if (extension === ".png") {
    return bytes.length >= 8
      && [137, 80, 78, 71, 13, 10, 26, 10].every((value, i) => bytes[i] === value);
  }
  if (extension === ".jpg" || extension === ".jpeg") {
    return bytes.length >= 3 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
  }
  return extension === ".webp"
    && bytes.length >= 12
    && String.fromCharCode(...bytes.slice(0, 4)) === "RIFF"
    && String.fromCharCode(...bytes.slice(8, 12)) === "WEBP";
}

function base64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

async function readLimitedBody(request: Request): Promise<Uint8Array> {
  const reader = request.body?.getReader();
  if (!reader) return new Uint8Array();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > MAX_REQUEST_BYTES) {
        await reader.cancel();
        throw new RangeError("Feedback request is too large.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const body = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

function base64Url(bytes: Uint8Array): string {
  return base64(bytes).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function encodeHeader(value: string): string {
  return `=?UTF-8?B?${base64(new TextEncoder().encode(value))}?=`;
}

function formatEmail(
  kind: string,
  user: { id: string; email: string; user_metadata?: Record<string, unknown> },
  message: string,
  attachment?: { filename: string; content: string; content_type: string },
): Uint8Array {
  const displayName = typeof user.user_metadata?.display_name === "string"
    ? user.user_metadata.display_name.replace(/[\r\n<>"]/g, "").trim().slice(0, 80)
    : "";
  const safeEmail = user.email.replace(/[\r\n<>"]/g, "");
  const headers = [
    `To: ${FEEDBACK_TO}`,
    `From: AI Evidence Extractor <${FEEDBACK_TO}>`,
    `Reply-To: ${displayName ? `${encodeHeader(displayName)} ` : ""}<${safeEmail}>`,
    `Subject: [AI Evidence Extractor] ${kind}`,
    "MIME-Version: 1.0",
  ];

  if (!attachment) {
    headers.push("Content-Type: text/plain; charset=UTF-8", "Content-Transfer-Encoding: base64");
    const body = base64(new TextEncoder().encode(message));
    return new TextEncoder().encode(`${headers.join("\r\n")}\r\n\r\n${body}`);
  }

  const boundary = `aee_${crypto.randomUUID().replace(/-/g, "")}`;
  headers.push(`Content-Type: multipart/mixed; boundary="${boundary}"`);
  const bodyText = base64(new TextEncoder().encode(message));
  const attachmentLines = attachment.content.match(/.{1,76}/g)?.join("\r\n") || "";
  const safeFilename = attachment.filename.replace(/[^A-Za-z0-9._-]/g, "_");
  const body = [
    `--${boundary}`,
    "Content-Type: text/plain; charset=UTF-8",
    "Content-Transfer-Encoding: base64",
    "",
    bodyText,
    `--${boundary}`,
    `Content-Type: ${attachment.content_type}; name="${safeFilename}"`,
    "Content-Transfer-Encoding: base64",
    `Content-Disposition: attachment; filename="${safeFilename}"`,
    "",
    attachmentLines,
    `--${boundary}--`,
    "",
  ].join("\r\n");
  return new TextEncoder().encode(`${headers.join("\r\n")}\r\n\r\n${body}`);
}

async function gmailRequest(
  path: string,
  accessToken: string,
  body?: Record<string, string>,
): Promise<Response> {
  return await fetch(`https://gmail.googleapis.com/gmail/v1/users/me/${path}`, {
    method: body ? "POST" : "GET",
    headers: {
      authorization: `Bearer ${accessToken}`,
      ...(body ? { "content-type": "application/json" } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
}

Deno.serve(async (request: Request) => {
  if (request.method !== "POST") {
    return jsonResponse(405, { error: "Method not allowed." });
  }

  const contentLength = Number(request.headers.get("content-length") || "0");
  if (contentLength > MAX_REQUEST_BYTES) {
    return jsonResponse(413, { error: "Feedback request is too large." });
  }

  const authorization = request.headers.get("authorization") || "";
  const accessToken = authorization.match(/^Bearer\s+(.+)$/i)?.[1];
  const supabaseUrl = Deno.env.get("SUPABASE_URL");
  const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const gmailClientId = Deno.env.get("GMAIL_CLIENT_ID");
  const gmailClientSecret = Deno.env.get("GMAIL_CLIENT_SECRET");
  const gmailRefreshToken = Deno.env.get("GMAIL_REFRESH_TOKEN");
  if (!accessToken) {
    return jsonResponse(401, { error: "Sign in before sending feedback." });
  }
  if (!supabaseUrl || !serviceRoleKey || !gmailClientId
      || !gmailClientSecret || !gmailRefreshToken) {
    console.error("Feedback relay is missing required server configuration.");
    return jsonResponse(503, { error: "Feedback email is not configured." });
  }

  let user: { id?: string; email?: string; user_metadata?: Record<string, unknown> };
  try {
    const authResponse = await fetch(`${supabaseUrl}/auth/v1/user`, {
      headers: {
        apikey: serviceRoleKey,
        authorization: `Bearer ${accessToken}`,
      },
    });
    if (!authResponse.ok) {
      return jsonResponse(401, { error: "Your sign-in has expired. Sign in again." });
    }
    user = await authResponse.json();
  } catch (error) {
    console.error("Feedback relay could not verify the signed-in user.", error);
    return jsonResponse(502, { error: "Could not verify your account." });
  }
  if (typeof user.id !== "string" || typeof user.email !== "string") {
    return jsonResponse(401, { error: "A valid signed-in app account is required." });
  }

  let form: FormData;
  try {
    const body = await readLimitedBody(request);
    form = await new Request(request.url, {
      method: "POST",
      headers: request.headers,
      body,
    }).formData();
  } catch (error) {
    if (error instanceof RangeError) {
      return jsonResponse(413, { error: "Feedback request is too large." });
    }
    return jsonResponse(400, { error: "Feedback form data is invalid." });
  }
  const kind = form.get("kind");
  const message = form.get("message");
  if (typeof kind !== "string" || !ALLOWED_CATEGORIES.has(kind)
      || typeof message !== "string" || message.trim().length < 10
      || message.trim().length > 3000) {
    return jsonResponse(400, { error: "Feedback details are invalid." });
  }

  const uploaded = form.get("attachment");
  let attachment: { filename: string; content: string; content_type: string } | undefined;
  if (uploaded !== null) {
    if (!(uploaded instanceof File) || uploaded.size < 1
        || uploaded.size > MAX_ATTACHMENT_BYTES) {
      return jsonResponse(400, { error: "The screenshot must be 5 MB or smaller." });
    }
    const extension = extensionOf(uploaded.name);
    const contentType = MIME_TYPES[extension];
    const bytes = new Uint8Array(await uploaded.arrayBuffer());
    if (!contentType || !hasValidImageSignature(extension, bytes)) {
      return jsonResponse(400, { error: "Attach a valid PNG, JPG, or WebP screenshot." });
    }
    attachment = {
      filename: uploaded.name.replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 120) || "screenshot",
      content: base64(bytes),
      content_type: contentType,
    };
  }

  try {
    const rateLimitResponse = await fetch(`${supabaseUrl}/rest/v1/rpc/claim_feedback_slot`, {
      method: "POST",
      headers: {
        apikey: serviceRoleKey,
        authorization: `Bearer ${serviceRoleKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ p_user_id: user.id }),
    });
    if (!rateLimitResponse.ok) {
      console.error("Feedback rate-limit check failed.", rateLimitResponse.status);
      return jsonResponse(503, { error: "Feedback is temporarily unavailable." });
    }
    if (await rateLimitResponse.json() !== true) {
      return jsonResponse(429, { error: "Please wait a moment before sending another message." });
    }
  } catch (error) {
    console.error("Feedback rate-limit check failed.", error);
    return jsonResponse(503, { error: "Feedback is temporarily unavailable." });
  }

  try {
    const tokenResponse = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        client_id: gmailClientId,
        client_secret: gmailClientSecret,
        refresh_token: gmailRefreshToken,
        grant_type: "refresh_token",
      }),
    });
    if (!tokenResponse.ok) {
      console.error("Google rejected the configured Gmail refresh token.", tokenResponse.status);
      return jsonResponse(502, { error: "The feedback email could not be delivered." });
    }
    const tokenData = await tokenResponse.json();
    if (typeof tokenData.access_token !== "string") {
      console.error("Google token response did not contain an access token.");
      return jsonResponse(502, { error: "The feedback email could not be delivered." });
    }

    const emailResponse = await gmailRequest("messages/send", tokenData.access_token, {
      raw: base64Url(formatEmail(
        kind,
        user,
        [
          `Type: ${kind}`,
          `From: ${user.email}`,
          `Account ID: ${user.id}`,
          "",
          message.trim(),
        ].join("\n"),
        attachment,
      )),
    });
    if (!emailResponse.ok) {
      let payload: unknown;
      try {
        payload = await emailResponse.json();
      } catch {
        payload = null;
      }
      const error = payload && typeof payload === "object" && "error" in payload
        && payload.error && typeof payload.error === "object"
        ? payload.error as Record<string, unknown>
        : {};
      const reasons = Array.isArray(error.errors)
        ? error.errors.flatMap((item) => {
          if (!item || typeof item !== "object" || !("reason" in item)) return [];
          const reason = item.reason;
          return typeof reason === "string" && /^[A-Za-z0-9_.-]{1,80}$/.test(reason)
            ? [reason]
            : [];
        })
        : [];
      console.error("Gmail API rejected the feedback message.", JSON.stringify({
        httpStatus: emailResponse.status,
        apiCode: typeof error.code === "number" ? error.code : undefined,
        apiStatus: typeof error.status === "string"
          && /^[A-Za-z0-9_.-]{1,80}$/.test(error.status)
          ? error.status
          : undefined,
        reasons: [...new Set(reasons)],
      }));
      return jsonResponse(502, { error: "The feedback email could not be delivered." });
    }
  } catch (error) {
    console.error("Gmail API request failed.", error);
    return jsonResponse(502, { error: "The feedback email could not be delivered." });
  }

  return jsonResponse(200, { sent: true });
});
