"""Common functions."""
import torch


def generate_response_fn(args, utt, persona=''):
	"""Use appropriate model to generate a response."""
	if args.model_type == 'blender':
		args.model.reset()
		if persona.strip():
			utt = persona.strip() + '\n' + utt
		args.model.observe({'text': utt, 'episode_done': False})
		response = args.model.act()
		return response['text']
	elif args.model_type == 'dialogpt':
		if persona.strip():
			input_ids = args.tokenizer.encode(persona + args.tokenizer.eos_token + utt + args.tokenizer.eos_token)
		else:
			input_ids = args.tokenizer.encode(utt + args.tokenizer.eos_token)
		input_id_len = len(input_ids)
		input_ids = torch.tensor(input_ids, device=args.device, dtype=torch.long).unsqueeze(0)
		# Hardcoded params for generation for now.
		out = args.model.generate(input_ids, max_length=40, temperature=0.7, top_k=40,
								  device=args.device, do_sample=True)
		text = args.tokenizer.decode(out[:, input_id_len:][0], skip_special_tokens=False)
		if text.find(args.tokenizer.eos_token) > 0:
			text = text[:text.find(args.tokenizer.eos_token)]
		text = trim_text(text)
		text = text.strip()
		print('DialoGPT: {}'.format(text))
		return text


def trim_text(text):
	"""Helper fn to cut off generated output at the first ./?/! if there is one."""
	end_punc = '.!?'
	min_end_idx = 1000
	for end in end_punc:
		end_idx = text.find(end)
		if 0 < end_idx < min_end_idx:
			min_end_idx = end_idx
	if min_end_idx == 1000:
		return text
	else:
		if min_end_idx + 2 < len(text) and text[min_end_idx + 1] == '"':
			return text[:min_end_idx + 2]
		else:
			return text[:min_end_idx + 1]
