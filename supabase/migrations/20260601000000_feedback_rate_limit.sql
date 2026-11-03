create table if not exists public.feedback_rate_limits (
  user_id uuid primary key references auth.users(id) on delete cascade,
  last_sent_at timestamptz not null
);

alter table public.feedback_rate_limits enable row level security;
revoke all on table public.feedback_rate_limits from public, anon, authenticated;
grant all on table public.feedback_rate_limits to service_role;

create or replace function public.claim_feedback_slot(p_user_id uuid)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  claimed boolean;
begin
  insert into public.feedback_rate_limits (user_id, last_sent_at)
  values (p_user_id, statement_timestamp())
  on conflict (user_id) do update
    set last_sent_at = excluded.last_sent_at
    where public.feedback_rate_limits.last_sent_at
      <= statement_timestamp() - interval '60 seconds'
  returning true into claimed;

  return coalesce(claimed, false);
end;
$$;

revoke all on function public.claim_feedback_slot(uuid) from public, anon, authenticated;
grant execute on function public.claim_feedback_slot(uuid) to service_role;
