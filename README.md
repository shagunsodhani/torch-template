# torch-template
Boiler plate code for Torch based ML projects

## Parsing Logs

* The generated log file contains a json string per line. It can be grepped using [jq](https://stedolan.github.io/jq/).
* For example, to grep the `mode` and the `loss` fields in the log files, run `cat log.txt | jq --compact-output '[.mode, .loss]'`.

