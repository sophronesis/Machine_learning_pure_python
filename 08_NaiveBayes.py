from ml_algo import NaiveBayesClassifier

def main():
	trainset = [
	["Trump elected, Hillary lost and some news from Syria",'politics'],
	["Hackers rewarded for web attacks",'politics'],
	["Vacuum's quantum effects detected",'science'],
	["New way to create graphene discovered, scientists say",'science'],
	["McDonald\'s to move non-US tax base to UK",'business'],
	["Sears and Kmart store closings and losses mount","business"],
	["Russian doping: McLaren report says more than 1,000 athletes implicated","sport"],
	["Sky Sports will continue to broadcast the All-Ireland Gaelic Football and Hurling Championships after securing a five-year deal with the GAA.","sport"]
	]
	testset = [
	"Syria got new weapons",
	"Tax laws enforced in California",
	"Information about recent graphene extraction method"
	]
	newsmodel = NaiveBayesClassifier()
	for text,classname in trainset:
		newsmodel.train(text,classname)
	for text in testset:
		classname,prob = newsmodel.predict_class(text)
		print("{} {:2.3} - {}".format(classname,prob,text))	

if __name__ == '__main__':
	main()






