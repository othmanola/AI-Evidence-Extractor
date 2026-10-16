# Email account setup

The app supports email/password sign-in, Google sign-in, or an explicit guest-mode
choice. The Account page includes profile settings and optional profile photos.
Accounts store only a display name and optional profile photo in Supabase. Source
videos, analysis history, generated reports, and results remain on the local
computer and are not uploaded or synchronized.

## 1. Create the Supabase project

Create a project at [supabase.com](https://supabase.com/). In **Project Settings >
API**, copy the project URL and the public **publishable** key (or legacy `anon`
key). Enter them in **Account > Connection setup** in the app. Never enter or
distribute a `service_role` key or any secret key in the desktop app.

## 2. Create the profile table and private photo bucket

Open the project's **SQL Editor** and run the SQL shown in the Account page. It
creates the `profiles` table, enables row-level security, and creates a private
`profile-photos` bucket. The policies restrict profiles and photo objects to the
signed-in user's own ID. The app limits profile photos to PNG, JPEG, or WebP
images up to 5 MB.

Equivalent SQL:

```sql
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text not null default '',
  avatar_path text,
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;
grant select, insert, update on public.profiles to authenticated;

create policy "Users can read their own profile"
  on public.profiles for select to authenticated
  using ((select auth.uid()) = id);
create policy "Users can create their own profile"
  on public.profiles for insert to authenticated
  with check ((select auth.uid()) = id);
create policy "Users can update their own profile"
  on public.profiles for update to authenticated
  using ((select auth.uid()) = id)
  with check ((select auth.uid()) = id);

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('profile-photos', 'profile-photos', false, 5242880,
        array['image/png', 'image/jpeg', 'image/webp'])
on conflict (id) do update set public = false, file_size_limit = 5242880,
  allowed_mime_types = excluded.allowed_mime_types;

create policy "Users can read their own profile photos"
  on storage.objects for select to authenticated
  using (bucket_id = 'profile-photos' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "Users can upload their own profile photos"
  on storage.objects for insert to authenticated
  with check (bucket_id = 'profile-photos' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "Users can update their own profile photos"
  on storage.objects for update to authenticated
  using (bucket_id = 'profile-photos' and (storage.foldername(name))[1] = (select auth.uid())::text)
  with check (bucket_id = 'profile-photos' and (storage.foldername(name))[1] = (select auth.uid())::text);
```

Run this SQL once per project. If you already have policies with the same names,
review them instead of creating duplicates.

## 3. Enable email/password accounts

In **Authentication > Sign In / Providers**, enable **Email**. Choose whether to
require email confirmation. If confirmation is enabled, a new user should open
the confirmation message, then return to the app and sign in with their email
and password. Configure a custom SMTP provider before distributing the app to
many users; Supabase's default email sender has limited delivery.

## 4. Enable Google sign-in (optional)

In **Authentication > Sign In / Providers**, enable **Google** and enter the
OAuth client ID and secret from your Google Cloud project. In Google Cloud,
add your Supabase project's callback URL
(`https://<project-ref>.supabase.co/auth/v1/callback`) as an authorized
redirect URI for the OAuth web client.

In the Supabase project's **Authentication > URL Configuration**, add
`http://127.0.0.1:*/oauth/callback` to the redirect URL allow list. The app
uses a temporary local callback port for each sign-in and the Google button
opens the system browser. Return to the app after Google confirms the sign-in.

## Guest mode and privacy

Guest mode requires no Supabase project and keeps app usage local. The guest
choice is saved on the computer so the app does not ask again at every launch.
When signed in, only the account display name and optional profile photo are
stored in the connected Supabase project; the email identity is managed by
Supabase Auth. Analysis history, source footage, reports, and analysis output
are never synced by this feature.
