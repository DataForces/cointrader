# AutoTraders

Python autotrader using RESTful API

## install api
sudo pip install git+https://github.com/bitbankinc/python-bitbankcc.git

## setup your key in config.json

{
	"bitbank-trade": {
		"key": "your-trade-key",
		"secret": "your-trade-passwd"
	},
	"bitbank-ref": {
		"key": "your-refer-key",
		"secret": "your-refer-passwd"
	},
	"bitbank": {
		"coins": ["mona", "xrp", "btc", "bcc"]
	}
}

## usage
- training mona coin
> sh trade-mona.sh

- training xrp coin
> sh trade-xrp.sh

