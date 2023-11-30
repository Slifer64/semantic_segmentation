#! /bin/bash

print_info_msg ()
{
  local msg="$1"

  echo -e '\033[1;36m'$msg'\033[0m'
}

print_fail_msg ()
{
  local msg="$1"

  echo -e '\033[1;31m'$msg'\033[0m'
}

check_success ()
{
  local n_args=$#
  local status=$1

  if [ $status -ne 0 ]; then
    echo -e '\033[1;31m'"Failure..."'\033[0m'
    exit
  fi
}

